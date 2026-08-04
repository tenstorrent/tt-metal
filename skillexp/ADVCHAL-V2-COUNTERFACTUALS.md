# advchal-v2 — counterfactuals: what the stage would have found if it were set differently

Each section changes **one stage setting** and reports what the corpus would have produced. Five settings;
three are decided by measurement on hardware, two by recomputation over the cells' own artifacts.

Companion to [`ADVCHAL-V2-EXPERIMENTS.md`](ADVCHAL-V2-EXPERIMENTS.md) (E1–E8). Numbering continues from
there.

| # | setting changed | what changes |
|---|---|---|
| **E9** | oracle = **absolute** vs a reference, not differential vs the incumbent | the largest unshipped win **ships**, and the *incumbent* turns out to be the inaccurate one |
| **E10** | screen by **shipped-op cliff**, not advisor-boundary value | the winning candidate goes from rank 2/2/2/**4-of-27** to **rank 1 in all four cells** |
| **E11** | capture **2 layers**, as the skill's own §2a recommends | **never tried: 23 of 23 reconciliations ran at 1 layer** while the trigger condition held in 8/8 |
| **E12** | attribute in-chain re-grids to a **second channel** | **48.5 %** of the corpus's shipped gain is invisible to its own ceiling; **64.2 %** including the unshipped wins |
| **E13** | drop the **first timed repeat** from the floor | **nothing** — v2 already fixed this; no verdict flips. A v1 fix that is no longer needed |

---

## E9 — if the oracle were absolute: the biggest win ships, and the incumbent is the inaccurate one

**Setting changed.** Action point A1: build an **absolute** oracle against a reference the change cannot
move, at the model's own bar, and measure the **incumbent** against it too. Ship if the candidate is within
the bar **and no worse than the incumbent**.

**Applied to** gemma-4-26B B's R=22 residual/norm candidate — the −12.44 %/layer win from
[E7](ADVCHAL-V2-EXPERIMENTS.md) whose correctness I had to leave unverified because gemma's real weights are
absent from this host.

**The reference used instead:** the model's own `FunctionalDecoder`, bfloat16 throughout, on the same
synthetic weights and identical inputs. The optimized path quantises dense/experts to `BFLOAT8_B`, so the
reference is strictly higher precision. Both configurations get their own fresh KV cache (`paged_update_cache`
writes in place).

| layer kind | R=0 — the frozen incumbent, **what shipped** | R=22 — the discarded candidate |
|---|---|---|
| **sliding attention** | **0.9834713471213876 — FAILS the 0.995 bar** | **0.9993133247174041 — passes** |
| full attention | 0.999421221208843 — passes | 0.999682936454395 — passes |

Bit-identical on a repeat run. And recall the differential number from E7: R=22 vs R=0 was **0.98322**,
which *fails* a 0.995 bar.

### The finding

**A differential oracle cannot tell you which side moved.** Here it flagged the candidate as suspect when
the *incumbent* is the outlier: against a higher-precision reference the candidate is **0.0159 closer**, and
on sliding attention the shipped configuration does not clear the model's own bar at all.

Under the differential rule the stage rejects R=22. Under A1 it ships it — **decisively**, on both speed and
accuracy.

That is now the **second** cell where the differential oracle flagged the better configuration
(phi FN was the first, [E1](ADVCHAL-V2-EXPERIMENTS.md)). In both cases the rule punished the change that was
closer to the reference.

⚠ **Caveat, stated.** These are synthetic weights. The cell's own real-weight oracle passed R=0 at decode
PCC 0.999499, so R=0 is not broken with real weights — the synthetic setup amplifies something specific to
the unsharded sliding path. What survives regardless, and is what A1's rule turns on, is the **ordering**:
R=22 is closer to the reference than R=0 on **both** layer kinds.

---

## E10 — if screening were ordered by the cliff instead of by boundary value

**Setting changed.** The stage screens "in the order the reconciliation gives", which ranks by
advisor-attributable **boundary** value. Action point C1b proposes ranking by **shipped op cost among ops on
≤2 cores where the advisor wants more**.

Recomputed over the corpus's own per-op data, for the four cells whose win turned out to be a low-core
reduction:

| cell | order A (the stage's): top 3 | rank of the winner | order B (cliff): top 3 | rank of the winner |
|---|---|---|---|---|
| g26 onA | linear 122 µs, **rms_norm 45**, rms_norm 44 | **2 of 13** | **rms_norm 45**, rms_norm 44, rms_norm 44 | **1 of 7** |
| g26 B | linear 123 µs, **rms_norm 44**, rms_norm 44 | **2 of 13** | **rms_norm 44**, rms_norm 44, rms_norm 44 | **1 of 7** |
| nm FN | linear 44 µs, **rms_norm 26** | **2 of 2** | **rms_norm 26** | **1 of 1** |
| phi FN | linear 104, linear 73, qkv-heads 57 | **4 of 27** | **rms_norm 45**, rms_norm 44 | **1 of 2** |

**The cliff order puts the winner first in all four cells**, and it shrinks the list the cell has to work
through — 27 candidates to 2 on phi FN, 13 to 7 on gemma.

phi FN is the instructive one: under the stage's order its winner sits **4th of 27**, behind three linear/DS
rows. Those rows are where its early screening effort went, and the stage recommends screening DS advice
**last** precisely because it never wins — yet boundary-value ordering puts it first.

---

## E11 — if the capture window were 2 layers: never tried, in any cell

**Setting changed.** `reconcile.py --layers-in-window N`. SKILL.md §2a, "Consider capturing more than one
consecutive layer", exists because the py↔IR transition pins every graph input and output to DRAM
interleaved, so at N=1 a layer's entry and exit edges are **capture artifacts**, not decoder behaviour.

**What the corpus did:**

| | |
|---|---|
| `layers_in_window` across **23** reconciliations (every cell, every layer kind) | **1 in 23 of 23** |
| `spill.ran` in the advisor's own report — the condition the skill cites for going to N=2 | **True in 8 of 8** cells checked |

So the recommendation is recorded, its trigger holds everywhere, and **it was followed nowhere**. The wording
is "consider" and the gate does not check it.

**What that leaves unresolved.** The `layer_handoff` diagnostic at N=1:

| pattern | count | what the note says |
|---|---|---|
| `entry_from_dram=True, exit_in_l1=True` | **13 of 23** | *"This layer loads its input from DRAM but leaves its output in L1"* — a real per-layer conversion, declared out of scope, never quantified |
| `entry_from_dram=True, exit_in_l1=False` | 8 of 23 | *"no layer-boundary DRAM round trip detected, **or the profile does not show it**"* — explicitly ambiguous |
| `entry_from_dram=False, exit_in_l1=True` | 2 of 23 | both llama cells |

**13 of 23 layer-kind runs flagged a layer-boundary cost that nobody measured, and all 23 ran at the one
window size where the question cannot be settled.** In the v1 corpus the same cost was measured at 33.6 µs
(phi) and 48.0 µs (gemma-12B) per model — real money, and the stage's position is that it belongs upstream.
That may be right, but at N=1 the stage cannot even tell whether it is there.

---

## E12 — if in-chain re-grids were a separate attribution channel

**Setting changed.** Price re-grids of ops that stay inside their L1 chain as a **second channel**. The
reconciliation ceiling counts only *boundary conversions the advice does not place*, so an in-chain re-grid
removes no boundary and prices at exactly **0.000 µs**.

Splitting every shipped win in the corpus by channel:

| cell | what shipped | channel | Δ model |
|---|---|---|---|
| phi arm onA | `rope_l1_rect32` (L1 residency) | 1 | −8.754 % |
| phi arm B | `rope_l1_chain` | 1 | −5.738 % |
| phi arm FN | rope query_key | 1 | −5.462 % |
| gemma-4-12B | Q+K+V+MLP residency + output chain | 1 | −2.236 % |
| gemma-4-26B FN | concat-heads → output projection | 1 | −2.036 % |
| gemma-4-26B B | `sliding_attention_o_chain` | 1 | −0.408 % |
| qwen FN | `packed_qkv_l1_chain` | 1 | −0.046 % |
| **gemma-4-26B onA** | **88-core RMSNorm re-grid** | **2** | **−13.006 %** |
| **north-mini FN** | **MoE RMSNorm re-grid to 32 cores** | **2** | **−10.226 %** |

| | cells | sum | mean per cell |
|---|---|---|---|
| channel 1 — boundary conversions, which the ceiling prices | 7 | −24.680 pp | −3.526 pp |
| **channel 2 — in-chain re-grids, which the ceiling prices at ZERO** | **2** | **−23.232 pp** | **−11.616 pp** |

**48.5 % of the corpus's total shipped improvement is invisible to the metric the stage uses to decide what
to screen.** A channel-2 win averages **3.3×** a channel-1 win.

Adding the four unshipped wins measured in E1/E4/E5/E7 — **all four are channel 2** (phi FN −8.5 pp,
g26 B −10.8 pp, g26 onA −0.7 pp, nm FN −1.06 pp):

> **channel 2 accounts for 64.2 % of everything this stage can deliver on this corpus.**

Two cells demonstrate the consequence directly: gemma-4-26B onA recorded a **0.000 µs** ceiling on both layer
kinds and shipped −12.98 % by screening anyway; gemma-4-26B B had the same signal and stopped at the ceiling.

---

## E13 — if the noise floor dropped the first timed repeat: nothing changes any more

**Setting changed.** v1's diagnosis was that the floors are mostly unfinished warm-up — one v1 harness did
**1** untimed replay and the first timed repeat carried 45–73 % of the whole spread in 4 cells. v2 raised
`WARMUP` to ≥10. Does dropping the first repeat still matter?

Recomputed from every v2 control's own `repeats_ms`:

| cell / control | floor µs | floor excl. 1st | ratio |
|---|---|---|---|
| **phi FN** | 1.064 | 0.701 | **1.52×** |
| g26 FN full-attention | 2.157 | 1.762 | 1.22× |
| nm onA | 1.841 | 1.508 | 1.22× |
| phi B | 0.713 | 0.694 | 1.03× |
| **the other 13 controls** | — | — | **1.00× — the first repeat is not the extreme** |

**13 of 17 controls show no effect at all, and no cell's `feasibility.verdict` flips.** phi FN's 1.52× is the
worst case and it was already `measurable` either way.

**This is a clean pass for v2**: the `WARMUP ≥ 10` rule did what it was written to do, and the v1 fix is no
longer needed. It also sharpens where the remaining floor lives — not in the first *repeat* but in the first
*process* ([E3](ADVCHAL-V2-EXPERIMENTS.md), 60×), which per-process warm-up cannot reach.

---

## What this set says overall

Ranked by what each setting was worth on this corpus:

1. **The oracle setting is worth more than everything else combined** (E9, E1). Two of the corpus's three
   largest wins were vetoed by a differential oracle that, when replaced with an absolute one, not only
   passes them but shows the *incumbent* to be the less accurate configuration.
2. **The screening-order setting costs nothing and would have found every big win first** (E10).
3. **The attribution setting decides what gets screened at all**, and it currently hides 48–64 % of the
   available value (E12).
4. **One recommended setting was never exercised by any cell** (E11) — because it says "consider" and
   nothing checks it.
5. **One v1 defect is genuinely fixed** and needs no further change (E13).
