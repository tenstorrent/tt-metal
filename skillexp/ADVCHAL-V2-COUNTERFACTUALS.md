# advchal-v2 — counterfactuals: what the stage would have found if it were set differently

Each section changes **one stage setting** and reports what the corpus would have produced. Ten settings:
five decided by measurement on hardware, four by recomputation over the cells' own artifacts, one by reading
the source. A scoreboard is at the end.

Companion to [`ADVCHAL-V2-EXPERIMENTS.md`](ADVCHAL-V2-EXPERIMENTS.md) (E1–E8). Numbering continues from
there.

| # | setting changed | what changes |
|---|---|---|
| **E9** | oracle = **absolute** vs a reference, not differential vs the incumbent | the largest unshipped win **ships**, and the *incumbent* turns out to be the inaccurate one |
| **E10** | screen by **shipped-op cliff**, not advisor-boundary value | the winning candidate goes from rank 2/2/2/**4-of-27** to **rank 1 in all four cells** |
| **E11** | capture **2 layers**, as the skill's own §2a recommends | **never tried: 23 of 23 reconciliations ran at 1 layer** while the trigger condition held in 8/8 |
| **E12** | attribute in-chain re-grids to a **second channel** | **48.5 %** of the corpus's shipped gain is invisible to its own ceiling; **64.2 %** including the unshipped wins |
| **E13** | drop the **first timed repeat** from the floor | **nothing** — v2 already fixed this; no verdict flips. A v1 fix that is no longer needed |
| **E14** | run the advisor at a higher **optimization level** | **nothing exists above 2** for layout advice — closed by reading the source |
| **E15** | stop screening **DS-matmul advice last** | a matmul candidate *did* win, and **65 % of matmul cost** is exempt from screening by the agreement clause (~5.0 ms) |
| **E16** | report the **best measured decoder**, not the advisor delta | the stage credited the advisor with **67 %** of what its own directions deliver (13.6 ms vs 20.2 ms) |
| **E17** | change the **batch** | **not testable** — the wins are batch-32-pinned by construction, and nothing records the dependency |
| **E18** | measure **two layers**, or measure **eagerly** | two layers are additive to ±1.8 %; and **eager measurement inverts every norm win** — the traced-replay rule is load-bearing |

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

---

## E14 — if the advisor were run at a higher optimization level: there isn't one

**Setting changed.** `ShardAdvisor(optimization_level=...)`; the stage uses **2**, and `ttnn-advise` exposes
`--opt-level` (default 2).

**Answer, from source — no run needed.** In `tools/ttnn-jit/_src/shard_advisor.py:166` the level does exactly
one thing for layout advice:

```python
mem_layout = "true" if self.optimization_level >= 2 else "false"
```

Beyond gating `memory-layout-analysis-enabled`, the level reaches only `workaroundOptions.optimizationLevel`
and a Conv3d config path. **Levels 2 and 3 are identical for sharding advice; ≤1 turns it off entirely.**

**The stage already uses the only setting that matters.** This line of inquiry is closed — the advice cannot
be improved by asking the advisor harder, only by changing its objective
([`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md) §7).

---

## E15 — if DS-matmul advice were not screened last

**Setting changed.** SKILL.md §4: *"Screen DS-matmul advice **last**. It has not won a measurement in this
corpus, and where it agrees with a shipped DS config there is nothing to screen."*

**Both halves are wrong for v2.**

**1. A matmul candidate did win.** Of 90 matmul/linear rows in the corpus, 26 are genuine screened
disagreements, and **one was kept**: gemma-4-12B full-attention `linear`, 129.4 µs, 12 → 55 cores. That cell's
shipped change (`Q+K+V+MLP` residency) is a change to linear ops. The "never wins" claim is v1-derived and
v2 contradicts it.

**2. Most DS cost was never screenable, so "it never wins" is self-fulfilling.** Classifying every
matmul/linear row by cost:

| classification | rows | share of matmul cost |
|---|---|---|
| **grid differs from shipped, but NO verdict** — the DS-family agreement clause | **55** | **64.7 %** (4,993 µs) |
| grid differs, screened and rejected | 25 | 27.1 % |
| exact grid agreement | 7 | 7.3 % |
| grid differs, screened and **kept** | 1 | 1.7 % |

Matmul/linear rows are **62.3 % of the profiled window on average** and up to **89.8 %** (qwen B linear
attention). Of that, roughly two-thirds sits in rows where **the shipped and advised grids differ but the tool
records agreement because both are DRAM-sharded** — the level-3 rule in `LayoutScore`
([`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md) §2). Nobody ever screened any of it.

**So the stage tells cells to deprioritise the ops carrying two-thirds of the cost, and its accounting exempts
two-thirds of those from screening in the first place.** Action point C5 (`agreed_on: grid | ds_family`) is
what makes that ~5.0 ms visible.

---

## E16 — if the stage reported the best measured decoder instead of the advisor delta

**Setting changed.** The stage's headline is *contribution* — incumbent minus candidate, with the incumbent
frozen. What would "best decoder reachable from the advisor's own directions" have been?

| cell / kind | incumbent | shipped | best measured | stage says | best says |
|---|---|---|---|---|---|
| phi FN dense | 0.808757 | 0.769096 | **0.700431** | −4.90 % | **−13.39 %** |
| g26 B sliding | 1.258327 | 1.254000 | **1.101768** | −0.34 % | **−12.44 %** |
| g26 onA sliding | 1.823508 | 1.587511 | **1.574985** | −12.94 % | −13.63 % |
| nm FN sliding MoE | 0.577971 | 0.518022 | **0.512764** | −10.37 % | −11.28 % |
| g26 FN sliding | 1.341153 | 1.318449 | 1.316251 | −1.69 % | −1.86 % |
| phi A dense | 0.656989 | 0.607172 | 0.607172 | −7.58 % | −7.58 % |
| phi B dense | 0.788610 | 0.748458 | 0.748458 | −5.09 % | −5.09 % |
| llama-8B dense | 0.667737 | 0.667737 | 0.667737 | 0.00 % | 0.00 % |

| | summed model-level saving |
|---|---|
| what the stage shipped | **13,601 µs** |
| best measured on the same decoders | **20,225 µs** (1.5×) |

**The stage credited the advisor with 67 % of what the advisor's own directions could deliver.** The missing
33 % is not new ideas — it is the same directions at a different grid, or the same candidate past an oracle
that should have passed it.

---

## E17 — if the batch were different: the wins are batch-pinned by construction, and nothing records it

**Setting changed.** The stage pins batch (32 for dense cells, 1 for MoE). Does the norm win survive another?

| batch | control | norm 11 | rope+norm 11 |
|---|---|---|---|
| 8 | **fails at build** (`assert.hpp:104`) | fails | fails |
| **32** | 0.807203 | 0.746605 | **0.701036** |
| 64 | **fails**: `TT_FATAL: Unsupported input shape` | **fails**: `TT_FATAL: Shard height 32 must match physical height 64 for width sharded` | fails |

**Not testable — phi's decoder only runs at batch 32**, and the norm knob is *separately* pinned: its memory
config hardcodes a one-tile-row shard height,

```python
shape=(ttnn.TILE_SIZE, width_tiles * ttnn.TILE_SIZE)   # height = 32 = exactly one tile row
```

so at any batch that is not exactly one tile row the shard spec is invalid.

**The finding is the pinning itself.** The corpus's largest wins are **batch-32-specific by construction**,
and nothing in `final.json` records that dependency — a reader sees "−13.4 %/layer" with no indication that
it evaporates at batch 64. `decode_batch` is recorded; *the candidate's batch-shape assumption* is not.

(Incidental: the batch-32 control reproduced at 0.807203 here against 0.808757 sixteen hours earlier and the
cell's own 0.807152 — a ~1.6 µs spread across the whole session, which is the honest cross-session floor.)

---

## Counterfactual scoreboard

| setting | verdict | worth |
|---|---|---|
| oracle: absolute, not differential | **change it** — the current rule can reject the more accurate configuration | the two largest wins |
| screening order: cliff, not boundary value | **change it** — free, and ranks the winner 1st in 4/4 | found every big win |
| attribution: add the in-chain channel | **change it** — 48–64 % of deliverable value is priced at zero | decides what gets screened |
| capture window: N≥2 when spill fires | **change it** — 23/23 ran at N=1; the trigger held in 8/8 | unquantified layer-handoff cost |
| DS-matmul: screen last | **drop the rule** — a matmul candidate won, and 65 % of matmul cost is exempt from screening anyway | ~5.0 ms invisible |
| floor: drop the first repeat | **no action** — v2 fixed it; 13/17 unchanged, no verdict flips | nothing |
| advisor optimization level | **no action** — 2 is the only meaningful setting | nothing |
| batch | **record it** — the wins are batch-pinned and the pin is undocumented | correctness of every published delta |
| one isolated layer | **no action** — two consecutive layers are additive to ±1.8 % (eager; traced case open) | nothing |
| traced replay, not eager | **already right, and say why** — eager inverts every norm win in the corpus | the entire win class |

---

## E18 — if the stage measured more than one layer, or measured eagerly

Two settings at once, because one probe answers both.

**Setting 1.** The stage times **one isolated layer** — which gets all 110 cores and all of L1 to itself — and
multiplies by the layer count. Every model-level number in the corpus rests on that extrapolation.

**Setting 2.** The stage mandates **traced decode replay**. What if a cell timed eager execution instead?

**Probe.** Built two *real* north-mini layers in one process (layer 1 = sliding MoE, layer 4 = full MoE),
timed each alone and then both back to back, under the stage's block structure (10 untimed warm-ups, 5 timed
blocks, each the mean of 50 replays). Repeated at three norm settings.

### Additivity

| norm cores | layer 1 alone | layer 4 alone | sum | measured together | excess |
|---|---|---|---|---|---|
| 0 (frozen) | 0.976412 | 0.929511 | 1.905923 | 1.872189 | **−33.7 µs (−1.77 %)** |
| 0 (repeat) | 0.971305 | 0.926575 | 1.897880 | 1.905330 | **+7.5 µs (+0.39 %)** |
| 16 | 1.016869 | 0.992450 | 2.009318 | 1.998050 | **−11.3 µs (−0.56 %)** |
| 32 | 1.041693 | 0.974995 | 2.016689 | 2.027412 | **+10.7 µs (+0.53 %)** |

**Two consecutive layers cost the sum of the two measured alone.** The excess is within ±1.8 %, changes sign
between runs, and is far inside the 25–188 µs block spreads. **No contention penalty between consecutive
layers** — so per-layer → per-model multiplication is not introducing an error of a size that matters.

⚠ **Scope of that conclusion.** This probe runs **eager**, so host dispatch dominates and is trivially
additive. It is therefore weak evidence about the *traced* case, where device work dominates and L1
contention would show if it existed. Building a two-layer **trace** is beyond what any cell's harness
supports, so the traced-mode question stays open. What can be said is that nothing in the additivity data
suggests a problem, and the stage's own §6 arithmetic is not obviously wrong.

### The accidental finding: eager measurement inverts every norm win

Compare the same layer and config across the two execution modes:

| north-mini layer 1, sliding MoE | frozen (1-core norm) | 16-core norm | verdict |
|---|---|---|---|
| **traced replay** (the stage's protocol, cell's own harness) | 0.577971 | **0.512764** | **−65.2 µs — a win** |
| **eager** (this probe's loop) | 0.971305 | 1.016869 | **+45.6 µs — a regression** |

The sharded norm **adds ~46 µs of host dispatch** (`to_memory_config`, `sharded_to_interleaved`, an extra
program) and **saves ~65 µs of device time**. Under traced replay the host cost is captured once and replayed
away; only the device saving remains. Eagerly, the host cost is paid every call and swamps the saving.

**So a cell that timed eagerly would have rejected every single norm win in this corpus** — including the two
that shipped. The stage's insistence on traced decode replay is not a detail; it is what makes this entire
class of optimisation visible at all.

That also explains a puzzle in the corpus: several cells' `*_profile` measurements are consistently ~1–3 %
slower than their timed counterparts, and one cell found that trace-replay rows carry no host markers between
signposts. Host-side work and device-side work move in opposite directions for these candidates, so any
mixing of the two measurement modes will produce contradictory rankings.

**No change needed — this setting is already right.** Recorded because it is the one place where the stage's
strictness is load-bearing, and because the *reason* is not stated anywhere in the skill: the file justifies
tracing as "what production does", not as "the only mode in which a placement win is visible".
