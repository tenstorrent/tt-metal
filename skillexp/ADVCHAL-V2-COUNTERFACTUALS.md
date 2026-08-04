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
| **E19** | look for the **wrong op**, not the wrong layout | gemma-4-12B pays **23×** the corpus mean to concatenate heads because it calls a different TTNN op — a defect class the stage cannot express |
| **E20** | widen **matmuls**, the largest apparent pool | **dead end, and it retracts part of my own C5** — DS matmuls are bandwidth-bound; the advisor's direction is **+65 %** slower |
| **E21** | look at the **boundary** bucket my own dataset was dropping | **`retilize` is 76.5 % of all boundary cost** — 191 ms/model, **24.4 % of qwen B's decode time**, with the advisor's ceiling correctly at 0.000 µs |

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
| the question itself: layouts only | **widen it** — "wrong op" is invisible to a layout-diff stage | ≈2.4–2.6 ms/model in one cell |

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

### E18b — and the inversion separates the two attribution channels exactly

Repeated the mode comparison on a second model, phi-3.5 FN, dense layer, batch 32 — and measured *both* of its
candidate classes:

| candidate | traced replay | eager | class |
|---|---|---|---|
| frozen control | 0.807203 | 1.258255 / 1.261008 | — |
| **rope query_key** — *what shipped* | 0.769096 → **−38.1 µs, win** | 1.203638 → **−56.0 µs, win** | **channel 1**: removes conversions |
| **norm 11 cores** | 0.746605 → **−60.6 µs, win** | 1.332182 → **+72.6 µs, LOSS** | **channel 2**: in-chain re-grid |
| rope + norm 11 | 0.701036 → −106.2 µs, win | 1.310241 → +50.6 µs, loss | mixed |

**The split is exact:**

- **Channel 1 wins in *both* modes.** Keeping the rope chain L1-resident *removes* work — fewer conversions on
  the device and fewer programs on the host. Nothing about how you measure it can hide it.
- **Channel 2 wins *only* under traced replay.** Re-gridding an op inside its chain *moves* work: it adds host
  programs (`to_memory_config`, `sharded_to_interleaved`) to buy device parallelism. Traced replay pays the
  host cost once; eager pays it every call.

So the measurement mode is not a detail of protocol hygiene — **it is what decides whether channel 2 exists at
all**, and channel 2 is 48–64 % of everything this stage can deliver (§E12). Two models, same direction, same
magnitude class (+46 µs host on north-mini, +73 µs on phi).

That also explains a puzzle in the corpus: several cells' `*_profile` measurements are consistently ~1–3 %
slower than their timed counterparts, and one cell found that trace-replay rows carry no host markers between
signposts. Host-side work and device-side work move in opposite directions for channel-2 candidates, so any
mixing of the two measurement modes will produce contradictory rankings.

**No change needed — this setting is already right.** Recorded because it is the one place where the stage's
strictness is load-bearing, and because the *reason* is not stated anywhere in the skill: the file justifies
tracing as "what production does", not as "the only mode in which a placement win is visible".

---

## E19 — a defect class the stage is structurally blind to: the wrong op

Chasing the largest starved op in the corpus led somewhere the stage cannot look.

### The observation

gemma-4-12B's decode path spends **102.6 µs (7.79 % of its full-attention window) concatenating heads on
ONE core**, and 51.1 µs (4.26 %) on sliding. Verified in its own profile CSV: `NLPConcatHeadsDeviceOperation`,
**24 of 24 instances on 1 core**, ~102.6 µs each — for comparison the layer norms on that same single core
cost 9.2 µs, so this is **12× the most expensive other 1-core op**.

### Every head-concatenation in the corpus, side by side

| op | cells | rows | mean µs | max µs | shipped cores |
|---|---|---|---|---|---|
| **`concatenate_heads`** | **gemma-4-12B only** | 2 | **76.9** | **102.6** | **1** |
| `nlp_concat_heads_decode` | 13 others | 18 | **3.4** | 9.4 | 16 / 24 / 32 |

Batch-matched (batch 32): phi 4.7–4.8 µs, llama 3.9–5.6 µs, qwen 9.0–9.4 µs — all on 24–32 cores.
**gemma-4-12B pays 23× the corpus mean for the same logical operation, because it calls a different TTNN op.**

### I tried to fix it in place. Three attempts, three kernel walls

`nlp_concat_heads` is handed `ttnn.L1_MEMORY_CONFIG` (interleaved) and the *very next line* reshards the
result into `self.decode_o_input_memcfg`, which is already width-sharded. So the obvious change is to have
concat write straight into it:

| attempt | result |
|---|---|
| `memory_config=self.decode_o_input_memcfg` (width-sharded) | `RuntimeError: bad optional access` |
| `memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG` | `RuntimeError: bad optional access` |
| switch to `ttnn.experimental.nlp_concat_heads_decode` | `TT_FATAL: Input tensor must be sharded` |

The third wall is the informative one, and it chains into a constraint the corpus already documented
elsewhere: `nlp_concat_heads_decode` needs a **sharded input**, which means SDPA must emit a sharded output,
and two other cells recorded exactly `TT_FATAL: Sharded output not supported for GQA` when they tried that
(phi exp17, gemma-4-26B FN). So the peers that pay 3.4 µs get there by keeping the whole
SDPA → concat → O-projection run sharded; gemma-4-12B drops to interleaved at SDPA and cannot rejoin.

**Controls measured while probing** (unchanged code, batch 32): full attention 1.308639 / 1.308410 ms,
sliding 1.219387 / 1.219423 ms — reproducible to 0.2 µs, so the probe harness was sound and the failures are
the kernel's.

### Size of it — measured cost, estimated saving

| | |
|---|---|
| **measured** | 102.6 µs/layer (full, 8 layers) and 51.1 µs/layer (sliding, 40 layers) on 1 core |
| **peer cost, measured on 13 other cells** | 3.4 µs mean, 9.4 µs worst |
| **estimated saving if it reached peer cost** | 8 × ~97 + 40 × ~46 ≈ **2,600 µs/model** (~4.5 % of its 58,520 µs estimate) |
| conservative, at the *worst* peer (9.4 µs) | ≈ **2,400 µs/model** |
| what that cell actually shipped | **−666.8 µs/model (−1.14 %)** |

⚠ **The saving is an estimate under a stated assumption** — that a sharded concat would reach peer cost. I
could not demonstrate it; all three in-place routes hit kernel constraints. The *cost* and the *peer
comparison* are measured; the *saving* is not.

### Why this matters more than the number

**The stage cannot express this defect.** Its question is *"which conversions does the advisor's plan not
place?"* — a question about **layouts**. This is a question about **op selection**: the decoder calls
`concatenate_heads` where its peers call `nlp_concat_heads_decode`. The advisor advised **DRAM** for the op,
the reconciliation duly filed it under DRAM-advice, and nothing in the ranked worklist could have surfaced it.
Neither the ceiling, nor the cliff check (C1b — the op *is* on ≤2 cores, but the advisor does not want it
widened, so rule A filters it out), nor the oracle, nor the grid ladder.

It is also **3.9× larger than what that cell shipped**, and the cell in question is the corpus's most
thoroughly screened (28 measurements, the most of any cell).

**The lens that found it is not in the stage at all: compare the same logical operation across models.**
See [`ADVISOR-VALUE`](ADVCHAL-V2-ADVISOR-VALUE.md) §8.

---

## E20 — the largest apparent opportunity is not one: matmul widening is a dead end

**Where it came from.** Generalising the cliff check from "op on ≤2 cores" to *any* op the advisor wants
widened, and ranking by `us × (1 − shipped/advised)` — the fraction of cost that could parallelise away:

| op | rows | parallelisable µs | total µs | cells | ever measured |
|---|---|---|---|---|---|
| **`linear`** | **67** | **4,970** | 5,980 | 13 | 4 of 67 |
| `rms_norm` | 53 | 1,057 | 1,150 | 9 | yes |
| `multiply` | 1 | 160 | 182 | 1 | no |
| `add` | 1 | 132 | 156 | 1 | no |

`linear` dominates at **4.7× the norm**, always the same signature: **shipped 12 cores, advised 55–99**.

### Measured on north-mini, one knob at a time from the shipped default

| layer 0 — dense projections | median ms | vs default |
|---|---|---|
| default: **12-core DRAM-sharded matmul** | 0.172532 / 0.172563 | — |
| `dram_sharded_dense_decode=False`: wide L1-sharded — **the advisor's direction** | 0.284863 / 0.285367 | **+65.2 %** |

| layer 1 — MoE expert matmuls | median ms | vs default |
|---|---|---|
| default | 0.517963 / 0.518237 | — |
| gate_up 48 / down 64 | 0.522919 | +0.96 % |
| gate_up 16 / down 16 | 0.528120 | +1.96 % |
| gate_up 64 / down 64 | **illegal** — `N tiles 48 must divide num_cores=64` | — |

### Result: the metric was wrong, and the stage's exemption was right

**DS matmuls are DRAM-bandwidth-bound — core count is not the limiting resource**, so
`us × (1 − shipped/advised)` is meaningless for them. The 4,970 µs is not parallelisable cost; it is an
artefact of applying a reduction-shaped metric to a bandwidth-shaped op.

Corpus record plus these probes: **1 win in 7 measured matmul-widening candidates.** The single win
(gemma-4-12B `linear` 12→55, kept) is a real exception, so this is model-dependent — but the default of not
screening DS-family matmuls is a *good* default.

⚠ **This retracts part of my own action point C5.** I had recommended treating a `ds_family` match with a
different grid as a screenable candidate, which would have sent cells to spend device time on a 65 %
regression. **Keep the `agreed_on: grid | ds_family` field** — it is needed for legibility and it misled this
analysis once — but **drop the recommendation to screen those rows.**

---

## E21 — the biggest cost in the corpus is a layout crossing no one is looking at

Pursuing E20 exposed a bucket my own per-op dataset had been silently dropping: `boundary`. Recovered from
each cell's `reconciliation_*.json` `disagreements[]`:

| what the boundary ops are, corpus-wide | µs | share | rows |
|---|---|---|---|
| **`retilize`** | **4,114.5** | **76.5 %** | 47 |
| `reshape_view` | 635.4 | 11.8 % | 29 |
| `fill_pad` | 262.1 | 4.9 % | 5 |
| `copy` | 233.5 | 4.3 % | 2 |
| `l1_to_dram` | 59.2 | 1.1 % | 28 |
| `dram_to_l1` | 54.8 | 1.0 % | 35 |
| `l1_regrid` | 16.5 | 0.3 % | 11 |

`retilize` — the tile ↔ row-major crossing — is **76.5 % of all boundary cost**, and the corpus's own pricing
makes it the most expensive conversion class by far (6.7–10.0 µs each, against 1.4–1.9 for an L1 regrid).

### Where it is

| cell | kind | retilize µs/layer | share of layer | layers | **per model** | advisor ceiling |
|---|---|---|---|---|---|---|
| **qwen B** | **linear_attention** | **3,983.5** | **25.2 %** | 48 | **191,210 µs** | **0.000 µs** |
| phi FN | dense | 63.7 | 8.8 % | 32 | 2,038 µs | 71.637 |
| qwen FN | full_attention | 21.2 | 2.1 % | 16 | 339 µs | 34.282 |
| qwen B | full_attention | 20.5 | 1.6 % | 16 | 328 µs | 33.698 |
| g26 onA | sliding_attention | 12.8 | 0.7 % | 25 | 321 µs | 0.000 |

**191,210 µs/model is 24.4 % of qwen B's 783,981 µs full-model estimate** — larger than every shipped win in
the corpus combined (13,601 µs), by 14×.

### What the ops are

| op | µs each | cores | edge |
|---|---|---|---|
| `UntilizeWithUnpaddingDeviceOperation` ×3 | **819.4 / 819.2 / 819.1** | **109** | `add → rms_norm` |
| `TilizeWithValPaddingDeviceOperation` ×2 | **671.1 / 671.0** | **109** | `add → rms_norm` |
| `TilizeWithValPadding` ×2 | 69.5 | 110 | `add → rms_norm` |

**They are already on 109 of 110 cores.** This is not under-parallelisation — it is pure layout-crossing cost,
paid every layer, and the decoder's own source says why:

> *Conv, reshape, and recurrent composite kernels currently require interleaved tensors; cross that boundary
> once after the packed projection instead of four times before four independent matmuls.*

The decoder has **already minimised the number of crossings**. What remains is the crossing itself, forced by
the conv/recurrent kernels demanding row-major.

### Why the whole exercise cannot see it

**The advisor's ceiling for that layer kind is 0.000 µs, and that is correct.** The ceiling counts *conversions
the advice does not place* — and the advisor's plan places these too, because they are legally required. So the
stage's honest answer is "the advisor can remove none of this", and it filed the cost under `boundary`:
reported, out of scope, uncredited.

That answer is right and the number is enormous. **It is not a placement problem at all** — it is a kernel
support gap (tiled input for the conv/recurrent composites), sitting at 24 % of a 27B model's decode time.

⚠ **Caveats.** Measured from qwen B's own reconciliation and its own 15,833 µs/layer window (which matches its
harness median of 15.85 ms, so these are per-layer figures, not replay aggregates). The per-model figure is the
stage's own linear extrapolation. And qwen **FN** almost certainly carries the same cost but it is
*unmeasured* — that arm's linear kind was declared tracer-unreachable, so no reconciliation exists for it.

### Answer to "are there more ops that could benefit more from the advisor?"

**From the advisor: no.** After matmuls are ruled out (E20) and `nlp_create_qkv_heads_decode` is shown to be a
batch artefact (§E19), the actionable `chain` pool is 5,067 µs corpus-wide, of which `rms_norm` (26.6 %) is the
proven class and the rest is small: `multiply` 9.0 %, `add` 4.8 %, `slice_static` 3.1 %, `rotary_embedding`
2.7 %, `concat` 2.0 %.

**Outside the advisor: yes, and it is much larger.** Ranked:

| # | opportunity | scale | kind of fix |
|---|---|---|---|
| 1 | `retilize` on qwen's `add → rms_norm` edge | **191 ms/model, 24.4 %** | tt-metal: tiled input for conv/recurrent composites |
| 2 | qwen's untraced linear attention | 97 % of decode time unexamined | tt-metal: tracer support for mutable-state `ttnn.copy` |
| 3 | `concatenate_heads` wrong-op in gemma-4-12B | ≈2.6 ms/model | tt-metal: sharded GQA SDPA output, then a decoder change |
| 4 | the four unshipped placement wins | ≈8 ms/model total | stage: oracle rule + grid ladder |
