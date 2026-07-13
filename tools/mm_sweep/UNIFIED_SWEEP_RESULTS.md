# Unified (Ns,Pk,Sm) regime-A BW-util sweep (2026-07-09, BH p150b)

Metric: DRAM BW-util% = `2·(MK+KN+MN) / kernel_time / 512 GB/s`. Timing = max-core all-RISC KERNEL
cycles @1.35 GHz, best of 6 runs (cold skipped). Configs chosen DIVISIBLE (no padding) so BW-util
reflects real traffic. kb = Kt_local/8 (ring-max deep block). Ring all-gather in0. All PCC-correct.
Harness: `tools/mm_sweep/unified_sweep.py` (SIGTERM timeout, per-job reset — no kill-9). 30 runs, 0 hangs.

## Best config per shape

| shape (M×K×N) | Mt | AI | best (Ns,Pk,Sm,kb,nsb) | cores | µs | GB/s | BW% |
|---|---|---|---|---|---|---|---|
| 32×4608×2304  | 1 | 31 | (1,3,1,6,3)   | 24 | 46  | 473 | **92** |
| 32×6144×9216  | 1 | 32 | (2,6,1,4,9)   | 96 | 267 | 429 | **84** |
| 32×6144×4608  | 1 | 32 | (1,8,1,3,9)   | 64 | 137 | 419 | **82** |
| 32×2048×2048  | 1 | 31 | (1,4,1,2,8)   | 32 | 22  | 398 | **78** |
| 64×6144×4608  | 2 | 62 | (1,12,1,2,18) | 96 | 150 | 387 | 76 |
| 64×6144×1536  | 2 | 61 | (1,6,2,4,6)   | 96 | 54  | 368 | 72 |
| 64×2048×6144  | 2 | 61 | (1,8,1,1,24)  | 64 | 80  | 329 | 64 |
| 128×2304×6144 | 4 | 119| (1,3,1,3,12)  | 24 | 92  | 330 | 64 |
| 128×6144×4608 | 4 | 122| (1,6,2,4,9)   | 96 | 183 | 325 | 64 |
| 128×6144×2304 | 4 | 119| (1,6,2,4,9)   | 96 | 99  | 308 | 60 |
| 256×4608×4608 | 8 | 230| (1,6,2,3,9)   | 96 | 147 | 320 | 63 |
| 256×6144×4608 | 8 | 233| (1,6,2,4,9)   | 96 | 199 | 312 | 61 |
| 512×3072×6144 | 16| 410| (1,6,2,2,6)   | 96 | 177 | 267 | 52 |
| 512×6144×4608 | 16| 429| (1,6,2,4,6)   | 96 | 255 | 265 | 52 |
| 512×6144×2304 | 16| 392| (1,6,2,4,3)   | 96 | 164 | 225 | **44** |

## Aggregate by Mt (geomean BW-util)

| Mt | BW-util | status |
|---|---|---|
| 1  | ~84% (78–92) | **SETTLED** — K-split ring hits 400–473 GB/s |
| 2  | ~70% (64–76) | mostly settled; shallow-K (K2048) weaker |
| 4  | ~62% (60–64) | plateau |
| 8  | ~62% (61–63) | plateau |
| 16 | ~49% (44–52) | **NEEDS DEBUGGING** |

## Findings

1. **Monotonic decline with Mt.** Mt=1 nails DRAM peak (K-split fills cores, in0 is a cheap k-slice).
   Mt=4–8 plateaus ~62%. Mt=16 drops to ~44–52% — the known large-Mt wall (in0 forwarding + reduction
   depth + compute overhead all scale with Mt while in1 read stays fixed).
2. **Pk6/Sm2 is the workhorse for Mt≥4.** Pk3/Sm4 (deeper reduction, more M-split) LOSES everywhere
   (~35–38%): the single m0 in1-forward reader serializes 3 unicasts (matches earlier "Sm4 overloaded";
   in1-mcast not wired into unified yet). Ns2 combos (~57%) also slightly worse than Pk6/Sm2.
3. **N size matters.** Small N amortizes the ring/reduction fixed cost poorly: 512×6144×**2304**=44% vs
   512×6144×**4608**=52%; 64×2048×6144 (shallow K) only 64%.
4. **Few-core K-split is efficient** when K is big: 32×4608×2304 = 473 GB/s (92%) on just 24 cores;
   128×2304×6144 = 64% on 24 cores (Pk3/Sm1) beats the 96-core Sm4 variant (38%).

## Where to debug next
- **Mt=16 (44–52%)** is the priority. Levers not yet in the unified path: in1-**mcast** to M-slaves
  (was 1.54× vs unicast in the standalone milestone — would lift Sm2/Sm4), and deep-K delivery (kb≥16
  blocked by the 8-bank ring's kb≤Kt_local/8 cap).
- **Small-N shapes** across all Mt — fixed-cost amortization.

---

## What the divisibility restriction cost, and the non-divisible corners (2026-07-09)

The first sweep only tested DIVISIBLE configs (to keep BW-util honest — padding inflates real traffic).
That excluded **762 / 1155 = 66%** of the `(Ns,Pk,Sm)` triples with ≤104 cores (per-shape 48–88% lost),
and collapsed `kb` to a single value (`Kt_local/8`). The whole point of first-class padding was these corners,
so I ran them (harness `unified_corners.py`; reports EFFECTIVE util on real bytes, DELIVERED BW on padded
bytes, and padding waste%):

| corner | shape | cfg (Ns,Pk,Sm,kb,nsb) | cores | waste% | eff% | delivered% | div-best% | Δ |
|---|---|---|---|---|---|---|---|---|
| Pk13 grid-fill | 32×6144×4608 | (1,13,1,1,9) | 104 | 8 | 79.9 | **86.6** | 82 | −2 |
| Pk11 grid-fill | 32×6144×4608 | (1,11,1,1,9) | 88 | 37 | 59.7 | 82.0 | 82 | −22 |
| Pk10 grid-fill | 32×6144×4608 | (1,10,1,2,9) | 80 | 66 | 51.6 | 85.9 | 82 | −30 |
| Sm3/Pk4 | 512×6144×4608 | (1,4,3,6,3) | 96 | 2 | 45.5 | 46.4 | 52 | −7 |
| Sm3/Pk4 | 256×6144×4608 | (1,4,3,6,6) | 96 | 1 | 50.0 | 50.6 | 61 | −11 |
| Sm3/Pk3 | 256×6144×4608 | (1,3,3,8,6) | 72 | 1 | 40.6 | 41.0 | 61 | −20 |
| Pk5/Sm2 | 512×6144×4608 | (1,5,2,1,6) | 80 | 4 | 40.0 | 41.6 | 52 | −12 |
| Sm3/Pk4 | 128×6144×4608 | (1,4,3,6,6) | 96 | 2 | 51.5 | 52.7 | 64 | −13 |

**Findings:**
1. **Padding works** — all corners PASS (correct) and run, none hang. The feature does what it was built for.
2. **Odd M-split (Sm=3) and non-divisor Pk (5) LOSE** by 7–20%, and it's NOT the padding (waste ≤4%): the
   configs are structurally worse. Sm=3's delivered% is also low (~40–50%), i.e. the single m0 in1-forward
   reader still serializes — confirms in1-mcast (not unicast) is the real missing lever, independent of Sm value.
3. **Grid-fill is the one genuinely interesting corner.** Pk13 → 104 cores delivers **86.6%** — the HIGHEST
   delivered BW of any Mt=1 config, beating divisible Pk8's 82%. More cores genuinely scale the ring engine.
   The only reason effective util lands at 79.9% (just under 82%) is the 8% K-padding waste.
4. **Padding waste is dominated by K rounding** `Kt_local=rup(cdiv(Kt,Pk), kb·8)` and is wildly config-dependent:
   Pk13/kb1 = 8% (fine), but Pk11 = 37%, Pk10 = 66% (catastrophic — `192/10=19.2` rounds up hard). So grid-fill
   only pays when Pk nearly divides Kt at kb=1. **Lever: grid-fill via the dimension with the CHEAPEST pad**
   (here N, not K) — pad N to add cores instead of paying K waste.

**Bottom line for the user:** the corners are now reachable and correct, but on these shapes the divisible
Pk6/Sm2 (Mt≥4) and high-Pk K-split (Mt≤2) remain best. The two real opportunities padding exposes: (a) grid-fill
to >96 cores (delivered BW rises to ~86%) if the pad is placed on a cheap dim, and (b) it makes in1-mcast the
clear next thing to wire in (Sm-corner value is capped by unicast serialization, not by divisibility).

---

## Mt=16 decomposition + run-down (2026-07-09) — 512×6144×4608, best cfg Pk6/Sm2 @ 260 GB/s (52%)

Ablation (each flag removes one cost; delta = its contribution). Timing = best-of-6 all-RISC KERNEL cyc.

| removed | GB/s | cost |
|---|---|---|
| — (baseline) | 260 | — |
| in0 read (--skipin0) | 280 | 7% |
| **in0 forward (--skipfwd)** | **337 (66%)** | **23% — dominant delivery cost** |
| reduction (--noreduce, Pk6) | +? | 8% |
| in1 read (--skipin1, Pk12/Sm1 proxy) | — | 8% |
| **compute-only floor (Pk12 skip-all)** | — | **182µs = 73% BW ceiling** |

**Reframe — Mt=16 is COMPUTE-BOUND in the pipeline:** compute-only floor 182µs > DRAM roofline 132µs.
Compute runs at only ~69% eff (159 TF/s = 1.66 TF/core vs ~2.4 peak) because **kb is capped at ≤4 by the
8-shard ring** (kb1→kb2 = 18% compute speedup; SP2 wants kb≥16). Pk12/Sm1 shifts the mix (reduction 22%,
in0-fwd 17%) — so Pk6/Sm2 already trades deep reduction for M-split and is well balanced.

### Corrected task order (evidence overturned my earlier guess)
1. ❌ **in1-mcast — DEMOTED.** in1 is only 8% at Sm2; it only bites at Sm≥3 (which lose anyway).
2. **Deep-K for compute efficiency** (the compute-bound ceiling). **DONE: unblocked deep-K in unified** —
   the padding hardwired the 8-ring (`Kt_local=rup(cdiv(Kt,Pk),kb·8)`), exploding Kt_local 8× for large kb.
   Made it ring-aware (`deepk ⇒ Kt_local=kb`, kb≥ceil(Kt/Pk)) + cut deep-K in1 cb depth 4→2. mshard kb=32
   now runs, PCC-correct. **But it TIES the ring (51% vs 52%)** — mshard baseline (350471 cyc) ≈ ring
   (350222): the deep-kb compute gain is offset by mshard's M-shard forward + deep in1 read. Negative result.
3. **in0-forward** (23%, biggest delivery cost, ceiling 66%) — no free win (chain/in0risc/in0order knobs <2%;
   group-mcast blocked by spread one-per-bank placement).

### Bottom line
Mt=16 sits at ~52%, ceiling ~66% (delivery) / ~73% (compute floor). The two real levers (deep-kb compute,
in0-forward) are both **coupled to the 8-ring**, and every incremental swap trades one cost for another
(deep-K removes ring-forward but adds M-forward+deep-in1; low Pk deepens reduction; in0direct adds 8× read).
Cracking Mt=16 needs a **delivery redesign that feeds compute deep-kb blocks AND avoids the in0 all-gather
forward simultaneously** — the "deep-K-capable delivery" open problem, not a knob/mode swap. Tooling now
supports deep-kb (env-gated --mshard/--in0direct in unified) for that work.

---

## Mt>=2 root-cause analysis + ranked next steps (2026-07-09)

**Root cause (measured):** in1-read volume is M-independent (K·N). Mt=1 hits 82-92% because all M-dependent
work is tiny and hides under the in1 read. For Mt>=2 the M-dependent work grows and breaks that overlap.
Ablation deltas (512/256/128/64 x 6144x4608, Pk6/Sm2 or Pk12/Sm1):

| Mt | baseline | in0 read | in0 fwd | reduction | compute-bound? |
|----|----------|----------|---------|-----------|----------------|
| 2  | 71-76%   | 1%       | 2%      | **10%**   | no (skipin1->179%, in1-read-bound) |
| 4  | 65%      | 2%       | 5%      | 6%        | no |
| 8  | 61%      | ~2%      | 6%      | ~6%       | borderline |
| 16 | 52%      | 7%       | **23%** | 8%        | **YES** (compute floor 182us > roofline 132us; kb<=4 -> 60-69% eff) |

Dominant M-dependent overhead SHIFTS: reduction (Mt2-4) -> in0-forward (Mt8-16) + compute-ineff (Mt16).

**Step 1 (heuristic, no kernel code): fill cores with N-slice, not K-split/M-split.** Ns adds NO reduction
(independent N cols) and NO in1-forward (reads own in1); Pk adds reduction (~output ~M), Sm adds in1-forward
(unicast serialization). Measured @96 cores, 6144x4608: Mt2 Pk6/Ns2 **80%** vs Pk12/Sm1 71%; Mt4 Pk6/Ns2 **73%**
vs Pk6/Sm2 65%; Mt8 Pk6/Ns2 ~63-65% vs 61%. Caveat: don't over-slice — Pk1/Ns9 = 59% (worst; low Pk -> deep
k-slice -> in0/cb0 balloon). Sweet spot: moderate Pk (~6) + Ns fill. Change the (Ns,Pk,Sm) picker accordingly.

**Step 2: tune nsb (large, non-monotonic, currently mis-set).** Same 256x6144x4608 Pk6/Ns2: nsb3=63%, nsb6=49%,
nsb9=65% — 16-pt swing. Auto-nsb (largest divisor<=budget) picks bad values. Sweep/model nsb per config.

**Step 3: trim L1 so Ns fits at higher Mt.** Mt8 Pk6/Ns2 OOM'd at nsb9 (fit nsb3); Mt16 Pk6/Ns2 OOM (M_block=16,
cb0=1MB). Reduce ring in1 cb depth (4->2/3), single-buffer where safe -> Ns usable at Mt8 w/ good nsb, + Ns+small-Sm at Mt16.

**Step 4 (hard, Mt16): deep-K-capable + forward-free delivery.** Mt16 is compute-bound (kb<=4 -> 60-69% eff; kb1->kb2
= 18% compute speedup; SP2 wants kb>=16) AND in0-forward is 23%. Ns can't rescue (OOM w/o Sm; Sm re-adds in1-fwd).
Deep-K (mshard kb=32) runs+correct but TIES (M-shard forward replaces ring forward). Need a delivery that feeds
deep-kb compute blocks AND avoids the in0 all-gather forward — both coupled to the 8-ring today. Real redesign.

---

## CORRECTION — genuine (Ns,Pk,Sm,kb,nsb) sweep (2026-07-09, 522 configs)

Earlier sections tested only 2-3 hand-picked (Ns,Pk,Sm) with HARD-CODED kb=Kt_local/8 and AUTO-nsb. That was
wrong; a real sweep (tools/mm_sweep/comprehensive_sweep.py) overturns several conclusions.

**Corrected achievable frontier (effective BW-util on real bytes, best config per shape):**

| shape | Mt | best (Ns,Pk,Sm) kb nsb | cores | BW% | previously claimed |
|---|---|---|---|---|---|
| 32x6144x4608  | 1 | (1,3,1) kb4 nsb6  | 24 | 95 | 82 |
| 64x6144x4608  | 2 | (1,12,1) kb2 nsb1 | 96 | **92** | 80 |
| 128x6144x4608 | 4 | (1,12,1) kb2 nsb1 | 96 | **89** | 73 |
| 256x6144x4608 | 8 | (1,12,1) kb2 nsb1 | 96 | **78** | 65 |
| 512x6144x4608 | 16| (1,12,1) kb2 nsb1 | 96 | **60** | 52 |
| 512x6144x2304 | 16| (1,12,1) kb2 nsb1 | 96 | 49 | 44 |

**What changed / retractions:**
1. **nsb is the dominant knob I was mis-setting.** For fixed (Ns,Pk,Sm,kb): 128x6144x4608 nsb1=89% / nsb6=77% /
   nsb18=61% — a 28-pt swing. nsb=1 (smallest N compute block) wins for K-split (tiny out/reduce blocks + max in1
   double-buffering). My auto-nsb picked large divisors = catastrophic.
2. **kb=2 wins for Mt>=2** (kb4 only for Mt=1 where compute is trivial). I had hard-coded kb=Kt_local/8 (=4 for Pk6).
3. **RETRACTED: "N-slice beats K-split" and "M-split helps".** Those were kb/nsb confounds. The universal best for
   Mt>=2 is PLAIN K-split (Pk12/Sm1) kb2 nsb1 — no N-slice, no M-split. Ns/Sm configs lose once nsb/kb are tuned.
4. Mt<=4 is essentially SETTLED (89-95%). Mt=8 good (78%).

**Decomposition at the TRUE-best config (Pk12/Sm1 kb2 nsb1):**
- Mt=8 256x6144x4608 (80%, roofline 121us): skipin1 -> 107% (113us). => **in1-read-bound**; compute+delivery
  floor (113us) is BELOW roofline; the 20% gap is imperfect in1<->compute overlap. Near-settled.
- Mt=16 512x6144x4608 (58-60%, roofline 132us): in0-read 5%, **in0-fwd 16%**, **reduction 12%**, in1 FULLY HIDDEN
  (5%). compute-only floor = **185us > roofline 132us => still COMPUTE-BOUND** (kb2 -> ~60-68% compute eff).

**Corrected next steps:**
- Mt<=4: settled. Mt=8: minor in1<->compute overlap gap (prefetch/buffering).
- Mt=16 (the real gap): compute floor 185us > roofline 132us => the CEILING-SETTER is compute efficiency, capped
  because kb<=ring-shard-limit. Needs deep-kb delivery (break the >=8-blocks-per-k-slice ring coupling). Secondary:
  reduction 12% (Pk12 linear depth-12 -> tree reduce) and in0-forward 16%. in1 is NOT the Mt=16 bottleneck.

---

## Mt=8/16 phased attack (2026-07-09) — in1+compute, reduction, in0, isolated

Best config for both: Pk12/Sm1 kb2 nsb1, 96c. Ablations (skip in0-read/fwd via --skipin0/--skipfwd,
reduction via --noreduce, in1 via --skipin1), best-of-5 KERNEL cyc.

### PHASE 1 — in1+compute floor (skip in0+fwd+reduce)
- **Mt=8** 256x6144x4608 (roofline 121us): compute-only=**75us** (62%); in1+compute=133us => **in1-READ-bound**
  (compute fully fits under in1; in1 read ~83% of peak). NOT compute-bound. Phase-1 goal already met.
- **Mt=16** 512x6144x4608 (roofline 132us): compute-only=in1+compute=**146us** (in1 fully hidden) => ~10% ABOVE
  BW floor. kb2 is a sharp optimum: kb1=218, kb4=273, deep-K (in0direct kb16/32/64)=196-337 — ALL worse
  (deep single-blocks don't pipeline in1<->compute; **deep-K rejected for compute**). Per-core ~2.07 TF/core
  (near peak). Only compute lever left = MORE CORES: 8-bank uses 96, 14 idle; 146x96/110=127us < roofline.
  => requires decoupling compute-core-count from the 8-bank delivery (converges with Phase 3).

### PHASE 2 — reduction (reintroduced)
Depth-bound linear chain, ~2us/level: Pk12 depth12=**23us** (10%), Pk6 depth6=10us, Pk8 depth8=9us. Lowering Pk
to cut reduction BACKFIRES (Pk6/Ns2 base 264us > Pk12 223us — in0 delivery grows). Fix: keep Pk12, **tree-reduce**
(depth 12->~4) => ~23us -> ~8us, save ~15us (~+3-4%). Tree-reduce reorders fp32 partial sums only — NO precision loss.

### PHASE 3 — in0 read + forward (reintroduced)
Mt16: in0 read=**11us** (5%), in0 **forward=36us (16%)** — the single biggest overhead. Forward is VOLUME-bound
(~M_block): Mt8 (M_block8)=16us, Mt16 (M_block16)=36us. in0[M,kslice] (512KB at Mt16) is shared by the 8 bank-cores;
ring reads-once (each 1/8) + 7 forward hops. in0direct (each core reads full slice, no fwd) = 8x redundant DRAM =
worse (rejected earlier). So read-once is right; the forward hops are the cost. Placement tension: the 8 cores
sharing a k-slice are bank-SPREAD (DRAM-read-optimal) => long forward hops. Levers to try: (a) forward-optimal
(clustered) placement for the sharing group vs DRAM-optimal — measure the trade; (b) decouple forward granularity
from kb (forward whole shard, fewer handshakes); (c) vary #cores reading in0.

### Combined Mt=16 budget (223us base, 60%): in1+compute 146 (66%) + in0-fwd 36 (16%) + reduction 23 (10%) + in0-read 11 (5%).
Priority: (1) more compute cores [Phase1+3, ~127us compute floor, biggest], (2) cut in0-forward 36us [Phase3],
(3) tree-reduce 23->8us [Phase2]. Mt=8: not compute/reduction bound — attack in1-read efficiency (83%->peak).

---

## PHASE 1 CORRECTION — swept (Ns,Pk,Sm) for the in1+compute FLOOR (2026-07-09)

Earlier Phase 1 narrowly fixed Pk12/Sm1 — wrong. Swept (Ns,Pk,Sm,kb,nsb) with in0-read+fwd+reduce ABLATED
(--skipin0 --skipfwd --noreduce) to find the slicing that minimizes in1+compute. The floor IS slicing-dependent:

| shape | best in1+compute floor | (Pk12/Sm1 gave) |
|---|---|---|
| Mt=8  256x6144x4608 | (2,6,1) kb1 nsb3 -> **121us = 100% roofline** | 133us (91%) |
| Mt=16 512x6144x4608 | (2,6,1) kb2 nsb1 -> **139us = 95%** | 146us (90%) |
| Mt=16 512x6144x2304 | (1,12,1) kb2 nsb1 -> 75us = 96% | (same) |

=> The compute floor is essentially AT the BW bound for the right slicing; NOT compute-bound, NOT needing deep-K
or more cores (those conclusions were Pk12-specific artifacts). kb1/nsb3 wins Mt8 (the "kb2 sweet spot" was also
Pk12-specific).

**But best-FLOOR != best-OVERALL.** Full decomposition Mt16 512x6144x4608:
| term | (2,6,1) best-floor | (1,12,1) best-overall |
|---|---|---|
| in1+compute | 139 | 146 |
| reduction | 10 (Pk6) | 24 (Pk12) |
| in0 read | 25 | 10 |
| **in0 forward** | **79** | 38 |
| total | 263us (50%) | 225us (58%) |

(2,6,1) has the better floor AND cheaper reduction but 2x the in0-forward (N-slice REPLICATES in0[M,kslice] across
Ns x more cores + its lower Pk => 2x deeper k-slice). That term alone is why Ns loses overall; Pk12 wins by
minimizing in0 (tiny k-slice) at the cost of deeper reduction.

**Reframed priority:** the whole gap for the best-floor config is in0 delivery (read 25 + fwd 79 = 104us of 263).
If Phase-3 makes in0 delivery cheap for N-slice (share in0 across the Ns cores instead of replicating), (2,6,1)
-> ~139+10+~20 = ~170us = ~68%, BEATING Pk12's 58%. So **in0-forward is THE lever**, and N-slice's replicated-in0
delivery is the specific target. Mt=8 floor is already 100% => Mt=8 is fully solvable by cheap in0 delivery alone.

---

## Shared-in0 delivery for N-slice — PROTOTYPED, NEUTRAL (2026-07-09)

Goal: make N-slice competitive by eliminating its redundant in0 delivery. In N-slice (Ns>1), the same
in0[M,k-slice] is needed by Ns*8 cores; today each of the Ns groups runs its own 8-bank ring => Ns x redundant
in0 read + Ns independent all-gathers. Env-gated `--in0share`: only the nn==0 LEADER of each (bank,k-slice) group
reads+rings in0; the nn>0 siblings skip the ring and get in0 from the leader. Kernel: in0_ring_writer.cpp
(compile args 20-22 in0share/share_valid/share_ready; per-core role runtime arg). PCC=1.0 all variants.

Three variants tried on (2,6,1) kb2 nsb1 (baseline Mt16=50%, Mt8=70%):
1. leader push, one-shot full forward after ring: Mt16 50->38%, Mt8 70->55% (REGRESS).
2. leader push, per-shard pipelined:            Mt16 50->39%, Mt8 70->57% (REGRESS).
3. receiver PULL, per-shard (leader only signals; idle receiver RISC copies): Mt16 50->49%, Mt8 70->68% (NEUTRAL).

Why neutral/regress: (a) leader-push concentrates forward work on the leader's writer RISC (7 ring-forwards + N
sibling-forwards) while receiver RISCs idle -> regress. (b) receiver-pull offloads to the idle RISC (fixes the
regress) but the total NoC DELIVERY work is unchanged: each receiver still pulls the full k-slice. The only thing
sharing removes is the redundant in0 DRAM READ (~13us) -- and that was NEVER the bottleneck (in1 read dominates
DRAM; in0 read is only 11-25us). The intrinsic cost is delivering in0[M,k-slice] onto Ns*8 cores over the NoC,
which sharing does not reduce.

CONCLUSION: N-slice's in0 disadvantage is fundamental (more cores need in0 + deeper k-slice), and shared delivery
does not fix it. **Pk12/Sm1 (max K-split => smallest per-core k-slice, NO N-replication) remains the best in0
strategy and the overall winner.** The real remaining in0 lever is on Pk12 itself: its 8-bank ring all-gather
forward = 38us/16% (Mt16), which looks latency-bound (7 sequential hops of small 256-tile slices). Attacking THAT
(fewer hops / better pipelining of the all-gather) is the next in0 target -- not read-dedup. Non-share path
unregressed (all changes gated by in0share).

---

## in0 all-gather latency attack (2026-07-09) — ring is near-optimal; cost is BW/congestion, not rounds

Target: Pk12/Sm1 (the winner) in0-forward = 38us/16% (Mt16), 16us (Mt8). Ring = G-1=7 sequential nearest-neighbor
rotations. Hypothesis: latency/round-bound => fewer rounds helps.

Prototyped `--in0scatter`: DIRECT scatter all-gather — each core reads its 1 shard, then writes it to all G-1
peers' same slot in ONE concurrent round (vs 7 sequential ring rounds). Same total bytes, 1/(G-1) the rounds.
PCC=1.0. Result:
| shape | ring (baseline) | +in0scatter |
|---|---|---|
| Mt16 512x6144x4608 | 224us (59%) | 240us (55%) REGRESS |
| Mt8  256x6144x4608 | 153us (79%) | 162us (75%) REGRESS |

=> Fewer rounds made it WORSE. The all-gather is **BW/congestion-bound, not round-latency-bound**: scatter bursts
G-1 concurrent writes from every core to bank-SPREAD peers (long hops + NoC congestion); the ring's nearest-neighbor
hops SPREAD OVER TIME avoid congestion. (Corroborated: forward scales super-linearly with shard size — Mt8 16us ->
Mt16 38us for 2x data.) The ring is already near-optimal for moving in0 around the 8 bank-spread cores; topology
changes (scatter, and by extension bidirectional) won't beat it.

REMAINING lever for the 38us: it's a STARTUP cost — compute stalls for it because IN0_KSLICE_RESIDENT waits for the
FULL k-slice before computing (needed to reuse in0 across the N_bpc=18 N-sub-blocks at nsb1). To HIDE it, stream the
FIRST N-sub-block's compute as the ring pushes k-blocks (while storing them resident for sub-blocks 2..N_bpc),
overlapping the all-gather under compute+in1-read. That's a compute.cpp (IN0_KSLICE_RESIDENT) change, not a delivery
change. Both --in0scatter and --in0share are env-gated; ring path unregressed (Mt16 224us = 59%).
