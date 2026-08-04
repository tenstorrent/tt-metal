# advchal-v2 — read this one file

**What was measured.** Stage 02b (`$advisor-challenger`) ran on 15 decoder cells to answer one question:
how much of a decoder's speed can the shard advisor be credited with, on a decoder already optimised
without it?

**The short answer.** The advisor's value is the size of the *placement defect it happens to find* — mostly
one class of defect, a reduction stuck on too few cores. Where that defect exists it is worth 6–13 % per
layer. Where it doesn't, the honest answer is zero, and 7 of 15 cells returned one.

**What the stage got wrong**, all measured on hardware afterwards (8 experiments):

1. It **discarded its own largest win** on a contradictory correctness rule — a candidate that is faster *and
   more accurate* than what shipped (−8.5 pp).
2. In **every** cell whose grid ladder had more than one legal rung, it **shipped the wrong rung**.
3. One cell **built a candidate, shipped it disabled, and never screened it** — it is worth **26× more than
   what that cell did ship**.

**Total left on the table from placement: ≈ 8.0 ms/model across four cells**, all identifiable by a **one-line
static check that needs no device time** (§3.8).

**And placement is not where the money is.** The corpus's single largest cost is **191 ms/model — 24.4 % of
qwen B's decode time — in tile ↔ row-major conversions** on one graph edge, 14× every shipped win combined. It is
a **shape choice in the decoder** (a 4-element conv window sitting on the 32-wide tile axis; the conversions run
at **~1 % of DRAM bandwidth**), it is fixable without touching tt-metal, and **no layout advisor could have
found it** because the fix is a graph rewrite, not a placement (§3.18–3.19).

**And the correctness objection to the biggest one does not survive contact with an absolute oracle.** Against
the model's own higher-precision reference, the discarded candidate scores **0.99931** and the configuration
that shipped scores **0.98347** — the *incumbent* is the one that fails the model's bar (§3.1).

**Bottom line across the eight cell/kinds I could re-measure: the stage shipped 13,601 µs/model of saving where
20,225 µs was reachable from the advisor's own directions — it credited the advisor with 67 % of what it found**
(§3.11).

---

## Where everything lives

| file | what's in it |
|---|---|
| **this file** | the account, and pointers |
| [`ADVCHAL-V2-IMPROVEMENTS.md`](ADVCHAL-V2-IMPROVEMENTS.md) | what to change — ideas, then action points |
| [`ADVCHAL-V2-EXPERIMENTS.md`](ADVCHAL-V2-EXPERIMENTS.md) | 8 experiments run on hardware to test the analysis |
| [`ADVCHAL-V2-COUNTERFACTUALS.md`](ADVCHAL-V2-COUNTERFACTUALS.md) | **10 stage settings changed one at a time** — what each would have found, with a scoreboard |
| [`ADVCHAL-V2-ADVISOR-VALUE.md`](ADVCHAL-V2-ADVISOR-VALUE.md) | **was the advisor necessary?** — detection, grid choice, hit rate, and what 7.4 h bought |
| [`ADVCHAL-V2-STAGE-ANALYSIS.md`](ADVCHAL-V2-STAGE-ANALYSIS.md) | the stage graded: what v2 fixed, 10 defects it kept |
| [`ADVCHAL-V2-ADVISOR-INTERNALS.md`](ADVCHAL-V2-ADVISOR-INTERNALS.md) | why the advisor advises what it does, from tt-mlir source + decision traces |
| [`ADVCHAL-V2-ORACLES.md`](ADVCHAL-V2-ORACLES.md) | every cell's correctness bar, and why they aren't comparable |
| [`ADVCHAL-V2-MEASUREMENTS.md`](ADVCHAL-V2-MEASUREMENTS.md) | all 149 harness measurements, per cell, in run order |
| [`ADVCHAL-V2-PER-OP.md`](ADVCHAL-V2-PER-OP.md) | every op the advisor placed differently |
| [`ADVCHAL-V2-PER-CELL.md`](ADVCHAL-V2-PER-CELL.md) | attribution accounting per cell |
| [`ADVCHAL-V2-RESULTS.md`](ADVCHAL-V2-RESULTS.md) | the headline table |
| `advchal-v2-narrative.json`, `advchal-v2-data.json` | machine-readable |

Everything is reconstructed from the cells' own session transcripts and artifacts, not their self-reported
summaries. Facts are sourced; where something is my inference it says so.

---

## 1. The method in six lines

1. **Freeze** the incoming decoder as the control — never re-tuned.
2. **Capture** the shipped graph, run the advisor, reconcile against a real op-level profile. The
   conversions the shipped decoder pays that the advice doesn't place = the **ceiling**.
3. **Screen** candidates on hardware: ≥10 warm-ups, 5 timed blocks, each the mean of ≥50 traced replays.
4. **Ship** only if *every* candidate block beats *every* control block, an oracle passes, and it
   re-confirms in a fresh process.
5. The **delta is the result**. Ties go to the incumbent. Zero is publishable.
6. Two numbers to read any cell by: the **noise floor** (control's block spread — 0.146 µs to 14.5 µs
   across cells) and the **band** (floor × layer count — a model gain smaller than its band isn't
   established).

---

## 2. The 15 cells

`FN` = `fuse-noadvise`, `B` = `nofuse-noadvise`, `onA` = `nofuse-noadvise-onA`. The `-onA` suffix says where
that cell's *incumbent* came from — **all 15 cells ran on the same host**, so no difference below is a
hardware difference.

| model | cell | control ms/layer | what shipped | model-level |
|---|---|---|---|---|
| llama-3.2-1B | exp17 | 0.3731 | nothing | **0.0 %** — honest zero |
| llama-3.1-8B | exp17 | 0.6650 | nothing | **0.0 %** — honest zero, re-verified |
| phi-3.5-mini | **A** | 0.6570 | `rope_l1_rect32` | **−8.75 %** |
| phi-3.5-mini | **B** | 0.7888 | `rope_l1_chain` | **−5.74 %** |
| phi-3.5-mini | **FN** | 0.8072 | rope only | −4.91 % — **−13.4 % measured and discarded** |
| phi-3.5-mini | exp17 | 1.1009 | nothing | 0.0 % — every direction overlapped or hard-failed |
| qwen3.6-27B | **FN** | 1.2083 full / 19.14 linear | `packed_qkv_l1_chain` | −445.7 µs — **inside its ±618.5 µs band** |
| qwen3.6-27B | **B** | 1.4494 full / 15.85 linear | nothing | 0.0 % — geometry hard-failed |
| gemma-4-12B | exp11 | 1.2541 / 1.3774 | `Q+K+V+MLP` + output chain | **−1.14 %** |
| gemma-4-26B | **B** | 1.2597 | `sliding_attention_o_chain` | **−147.9 µs** |
| gemma-4-26B | **onA** | 1.8252 | `advisor_norm88` | **−12.98 %/layer** |
| gemma-4-26B | **FN** | 1.3412 / 1.5394 | `advisor_concat_projection` | **−2.04 %** — *88-core norm regressed here* |
| north-mini | **FN** | 0.5537 MoE | MoE norm at 32 cores | **−10.23 %** — **16 cores is 1 pp better** |
| north-mini | **B** | 0.6138 / 0.2033 | nothing | 0.0 % — all geometries slower or stalled |
| north-mini | **onA** | 0.2918 / 0.8465 | nothing | 0.0 % — sparse MoE untraceable |

8 shipped, 7 returned zero. Of the zeros: **2 honest** (already well-placed), **3 structural** (the advisor
could not see the layer), **1 geometry wall**, **1 unseparable from its own noise floor**.

Per-cell narratives and every measurement: [`MEASUREMENTS`](ADVCHAL-V2-MEASUREMENTS.md).

---

## 3. The nineteen findings that matter

### 3.1 The corpus's largest win was measured, then discarded — by the stage's own rules

phi FN measured `rope + 11-core norm` at **−13.39 %** and shipped rope-only at −4.90 %.

I re-ran it. The discarded candidate is faster, deterministic, passes the model's own bar — and is **more
accurate against the HuggingFace reference than what shipped**:

| policy | speed | PCC vs HF reference (model's bar 0.995) |
|---|---|---|
| incumbent | 0.808757 ms | 0.99890 |
| shipped (rope only) | 0.769096 ms | 0.99890 |
| **discarded (rope + norm 11)** | **0.700431 ms** | **0.99904** ← better |

It was rejected for not being **bit-identical to the incumbent**: a differential oracle at bar `0.999999`
scored `0.9999910667` → `oracle_passed: false`, which the gate treats as a critical failure.

**This was not a cell error.** The skill says *"if a placement-only candidate moves PCC at all … reject the
candidate however fast it is"*, recommends a differential oracle, and separately warns that a differential
oracle against the frozen incumbent "cannot fail" — so the only reading that satisfies all three is a bar
at ≈1.0. Cost: **−3,466 µs/model available, −1,267 µs shipped.**

**And the rule is not applied consistently, because the oracle's *construction* decides the outcome:**

| cell | same class of change | how much it moved a differential PCC | oracle built | outcome |
|---|---|---|---|---|
| gemma-4-26B onA | 1 → 88 cores | **0.0177** | absolute, vs HuggingFace, bar 0.995 | **shipped** (−12.98 %) |
| phi-3.5 FN | 1 → 11 cores | **0.0000089** | differential, vs frozen incumbent, bar 0.999999 | **rejected** |

The cell whose change perturbed the output ~2,000× more shipped it. *(Not a controlled comparison — gemma's
figure is on synthetic weights, phi's on real, different models. The asymmetry in outcome doesn't depend on
that.)*

**And a differential oracle cannot tell you which side moved.** I built the absolute oracle A1 prescribes —
both configurations against the model's own bfloat16 `FunctionalDecoder`, same weights, same inputs — for
gemma-4-26B B's discarded candidate:

| layer kind | R=0 — **what shipped** | R=22 — **discarded** |
|---|---|---|
| sliding | **0.98347 — fails the 0.995 bar** | **0.99931 — passes** |
| full | 0.999421 | 0.999683 |

The differential number for the same pair is 0.98322, i.e. "the candidate moved". It did not: **the
incumbent is the outlier.** That is now twice that the differential rule flagged the configuration closer to
the reference. → [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E9

→ [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E1, [`STAGE-ANALYSIS`](ADVCHAL-V2-STAGE-ANALYSIS.md) §D1,
[`ORACLES`](ADVCHAL-V2-ORACLES.md)

### 3.2 One op class carries nearly all the value: a reduction on too few cores

| cell | norm | measured |
|---|---|---|
| gemma-4-26B onA | 1 → 88 cores | **−13.03 %/layer** |
| phi-3.5 FN | 1 → 11 cores | **−7.60 %/layer** (discarded, §3.1) |
| north-mini FN | 1 → 32 cores | **−10.37 %/layer** (16 is better still, §3.4) |

Every model *without* a low-core reduction returned zero from this family. Both llamas arrived with their
norm already on 32 cores.

### 3.3 The advised core count carries no throughput information

From the tt-mlir source at the pin: the advisor's objective is a **7-level lexicographic ordering with no
latency term at any level**, and core count is the **6th** tiebreaker — below "avoid one reshard". For
normalization ops the core-count term is overridden with the *input's* grid volume, so it cannot vary with
the candidate at all.

The advisor also **never executes anything** — it runs analysis-only against a mock device from a system
descriptor.

Verified in the decision traces: for llama's MLP norm, **32 and 64 cores were both valid candidates and
both lost to 22**, which matches neither its producer (64 cores) nor its consumer (90). Why 22 won is an
**open question** the traces don't answer — resolving it needs the pass rebuilt, which I didn't do.

→ [`ADVISOR-INTERNALS`](ADVCHAL-V2-ADVISOR-INTERNALS.md)

### 3.4 The best-swept cell still left ~1 pp, and the skill's rule pointed it the wrong way

The skill says *"never sweep only at or below an advised core count"*. north-mini's advised value was 22,
so it swept 22 / 32 / 64 and shipped 32. I swept the whole **legal** ladder:

| cores | 1 | 4 | 8 | 11 | **16** | 22 *(advised)* | **32** *(shipped)* | 64 |
|---|---|---|---|---|---|---|---|---|
| ms | 0.5780 | 0.5698 | 0.5182 | 0.5261 | **0.5128** | 0.5431 | 0.5180 | 0.5736 |

**The optimum is 16 — *below* the advised value** — worth a further 5.4–5.7 µs/layer on both MoE kinds
(≈ **−264 µs/model**) at unchanged accuracy, confirmed in interleaved fresh processes. The curve is
non-monotonic with a local *maximum* at the advised 22.

**The same is true of the corpus's biggest win.** gemma-4-26B onA shipped **88** cores; its own legal
ladder is `{11, 22, 44, 88}` and it measured only 88:

| cores | 1 *(frozen)* | 11 | 22 | **44** | **88** *(shipped)* |
|---|---|---|---|---|---|
| sliding ms | 1.8235 | 1.5736 | 1.5798 | **1.5750** | 1.5875 |
| full ms | 2.0127 | — | — | **1.7636** | 1.7760 |

**44 beats 88 by 12.2–12.4 µs/layer on both kinds** (≈ **−375 µs/model**), and is **bit-identical** to the
shipped configuration on sliding attention (PCC 1.0), so it inherits the oracle 88 already passed.

**Across the three cells with a low-core reduction, every one whose ladder had more than one legal rung
shipped the wrong rung.** Only phi — a plateau — happened to ship its best.

Two refutations of my own earlier claims: **phi** is a plateau from 11 to 48 cores (exact tile division buys
nothing), and north-mini's 40/44/48/55/88 are **illegal**, so its three points were nearly the whole legal
ladder above 22.

→ [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E2, §E4, §E5

### 3.5 Tracer coverage decides more than placement does

| model | unreachable | cause |
|---|---|---|
| qwen3.6-27B (both arms) | **48 of 64 layers** | linear attention's mutable-state `ttnn.copy` |
| north-mini onA | the whole sparse MoE tail | `ttnn.sparse_matmul` rejects tracer tensors |
| gemma-4-26B onA | **64.7 % sliding / 58.5 % full** of the window | sparse experts |
| phi-3.5 B | the fused-cache share | no tracer handler for `paged_fused_update_cache` |

Every structural zero in the corpus is a tracer-coverage zero. qwen's linear layers cost ~13× a full layer,
so **97 % of its model decode time** was never advised on. **This is a `$shard-advise`/tt-mlir coverage problem, not
a placement problem.**

### 3.6 A ~0 µs ceiling is not a stopping condition

The ceiling prices *boundary conversions the advice doesn't place*. A re-grid of an op that stays inside its
chain removes no boundary, so it prices at **0.000 µs**. gemma-4-26B onA recorded a 0.000 µs ceiling on both
layer kinds, screened the candidate anyway, and shipped **−12.98 %**. Two other cells trusted a similar
ceiling and shipped zeros.

### 3.7 The noise floor is mostly *between* processes — and more replays makes it worse

Two measured facts about the protocol, both surprising:

| | measured |
|---|---|
| The first harness process of a session recorded a floor of **11.838 µs**; the identical configuration later recorded **0.196 µs** | **60×**, from JIT-cache warmth *across* processes |
| Going from 250 replays per measurement (5 blocks × 50) to 1,800 (9 blocks × 200) | floor got **3–4× worse**: 0.4–0.7 µs → 1.3–3.0 µs |

The protocol justifies `ITERS ≥ 50` by "the spread between blocks is the spread of means, roughly
`sqrt(ITERS)` tighter". That holds only if the noise is i.i.d. within a run. It isn't — longer windows pick up
slow drift, and **drift does not average down**. 50 replays/block sits near the sweet spot; 200 is past it.

Consequence: the term worth attacking is the **cross-process** one, and per-process warm-up cannot touch it.
A cell whose control ran first carries an inflated floor, which directly changes its `feasibility.verdict`.

### 3.8 One static check, no device time, finds every big win in the corpus

Over the corpus's own per-op data, flag a cell when **an op's shipped grid is ≤2 cores, the advisor wants
strictly more, and the op is ≥2 % of the layer window**:

| flagged cell | largest actionable low-core op | what happened |
|---|---|---|
| gemma-4-26B onA | `rms_norm` 1→88c, 44.7 µs, 2.5 % | shipped −12.98 %/layer |
| **gemma-4-26B B** | `rms_norm` 1→88c, 44.5 µs, 3.7 % | **never screened — a −12.44 %/layer win was there** |
| north-mini FN | `rms_norm` 1→22c, 26.1 µs, 5.0 % | shipped −10.37 %/layer |
| north-mini onA | `rms_norm` 1→22c, 26.1 µs, 3.2 % | could not screen (untraceable) |
| phi-3.5 FN | `rms_norm` 1→11c, 44.5 µs, 6.1 % | measured −13.4 %, discarded on the oracle |
| *the other 9 cells* | *none* | **no double-digit layer win in any of them** |

**Every double-digit win in the corpus is in a flagged cell, and no unflagged cell produced one.** I used the
check to predict that gemma-4-26B B had an unscreened win, then measured it: the cell had **written** an
11- and 22-core residual/norm geometry, **shipped it disabled**, and never screened it. R=22 gives
**1.2583 → 1.1017 ms/layer (−12.44 %)** on sliding attention, reproduced four times, against floors of
0.4–4.2 µs. That is ≈ **−3,918 µs/model** versus the **−147.9 µs** it shipped.

**Its correctness is now settled** — see §3.1: against the model's own bfloat16 reference the candidate
scores 0.99931 and the shipped incumbent 0.98347, on both layer kinds the candidate is closer. (Synthetic
weights, since gemma's real ones are absent; what the ship rule turns on is the ordering, and that holds on
both kinds.)

→ [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md) §E7

### 3.9 Half of what the stage delivers is invisible to the metric it steers by

Splitting every shipped win by attribution channel — channel 1 = boundary conversions the ceiling prices,
channel 2 = re-grids of ops that stay inside their chain, which it prices at **0.000 µs**:

| | cells | sum of Δ model | mean per cell |
|---|---|---|---|
| channel 1 | 7 | −24.68 pp | −3.53 pp |
| **channel 2** | **2** | **−23.23 pp** | **−11.62 pp** |

**48.5 %** of the corpus's shipped improvement is channel 2. Add the four unshipped wins — all channel 2 —
and it is **64.2 % of everything this stage can deliver.** A channel-2 win averages **3.3×** a channel-1 win.

→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E12

### 3.10 One recommended setting was used by no cell at all

`layers_in_window` is **1 in 23 of 23** reconciliations — every cell, every layer kind — while `spill.ran`,
the condition SKILL.md §2a cites for going to 2 layers, is **True in 8 of 8** cells checked. The
recommendation is recorded, its trigger holds everywhere, and it was followed nowhere: the wording is
*"consider"* and the gate does not check it.

What that leaves open: **13 of 23** runs flagged *"this layer loads its input from DRAM but leaves its output
in L1"* — a real per-layer conversion, declared out of scope and never quantified — and the other 8 report
*"no round trip detected, **or the profile does not show it**"*. At a one-layer window the question is not
answerable, because the py↔IR transition pins both ends to DRAM by construction.

→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E11

### 3.11 The stage credited the advisor with two-thirds of what it actually found

Comparing what each cell shipped against the best configuration reachable from the advisor's *own* directions
on the *same* decoder (my re-measurements included):

| cell / kind | incumbent | shipped | best measured | stage says | best says |
|---|---|---|---|---|---|
| phi FN dense | 0.808757 | 0.769096 | **0.700431** | −4.90 % | **−13.39 %** |
| g26 B sliding | 1.258327 | 1.254000 | **1.101768** | −0.34 % | **−12.44 %** |
| g26 onA sliding | 1.823508 | 1.587511 | **1.574985** | −12.94 % | −13.63 % |
| nm FN sliding MoE | 0.577971 | 0.518022 | **0.512764** | −10.37 % | −11.28 % |
| g26 FN sliding | 1.341153 | 1.318449 | 1.316251 | −1.69 % | −1.86 % |
| phi A / phi B / llama-8B | — | = best | = best | — | nothing further found |

**Shipped 13,601 µs/model. Reachable 20,225 µs/model — 1.5×.** The missing third is not new ideas: it is the
same directions at a different grid, or the same candidate past an oracle that should have passed it.

### 3.12 Two rules that steer cells away from most of the cost

**Matmul/linear ops are 62.3 % of the profiled window on average** (up to 89.8 %), and the stage says to screen
DS-matmul advice **last** because "it has not won a measurement in this corpus". In v2 one *did* win
(gemma-4-12B, `linear` 12→55 cores, 129.4 µs, kept). And of all matmul cost:

| | rows | share of matmul cost |
|---|---|---|
| **grid differs from shipped, but recorded as agreement** (both DRAM-sharded) | **55** | **64.7 %** (≈5.0 ms) |
| screened and rejected | 25 | 27.1 % |
| exact agreement | 7 | 7.3 % |
| screened and **kept** | 1 | 1.7 % |

So two-thirds of the cost in the biggest op class is exempt from screening by the agreement clause, and the
ordering rule sends cells to it last anyway. → [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E15

### 3.13 One stage rule is doing more work than anyone said: traced replay

The stage insists on **traced decode replay**, justified in the skill as "what production does". Measured, it
is doing something much stronger — it is the only mode in which these wins exist at all:

| candidate | traced replay | eager | class |
|---|---|---|---|
| phi FN **rope only** — *what shipped* | −38.1 µs **win** | −56.0 µs **win** | **channel 1**: removes conversions |
| phi FN **norm 11 cores** | −60.6 µs **win** | **+72.6 µs LOSS** | **channel 2**: in-chain re-grid |
| north-mini **norm 16 cores** | −65.2 µs **win** | **+45.6 µs LOSS** | **channel 2** |

**The split is exact.** Channel 1 *removes* work, so it wins however you measure. Channel 2 *moves* work — it
buys device parallelism with extra host programs — so it wins only under traced replay, which pays the host
cost once instead of every call.

**A cell that timed eagerly would have rejected every channel-2 win in this corpus, including both that
shipped** — and channel 2 is 48–64 % of everything the stage can deliver (§3.9). Two models, same direction.

Related, and reassuring: two *real* consecutive layers cost the sum of the two measured alone to within
±1.8 % (sign-varying, inside the block spreads), so the per-layer → per-model multiplication is not
introducing an error that matters. *(Measured eagerly; the traced two-layer case needs harness support no cell
has.)*

→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E18

### 3.14 Zooming out: for this win class, the advisor was not necessary

Three independent tests of whether the advisor was needed for the wins it got credit for.

**Detection.** A rule using only the shipped profile — *op on ≤2 cores, ≥2 % of the layer* — flags 7 cells and
catches **all 4** win cells. Adding "and the advisor wants more cores" narrows it to 5 with the same recall. So
the advisor buys *precision*, not recall.

**Grid choice**, scored on the ladders I measured:

| selector | summed layer improvement over 3 cells | share of achievable |
|---|---|---|
| best legal grid (hindsight) | −32.61 pp | 100 % |
| **a fixed "closest to 16 cores" heuristic, no advisor** | **−32.42 pp** | **99.4 %** |
| the advisor's own recommended grid | −26.64 pp | **82 %** |

**Hit rate**, over all 118 per-op rows the corpus actually measured: **49 % — a coin flip.** By direction:
fewer cores 51 %, to DRAM 56 %, **more cores 11 % (1 won, 8 lost)**.

**And the reason that last number looks so bad is an accounting defect.** All 37 low-core `rms_norm` rows the
advisor wanted widened are recorded as `below_threshold` (30), `not_measurable` (5) or `rejected` (2) —
**`kept`: 0** — *including in the two cells that shipped exactly that change for −12.98 % and −10.23 %*. So the
one direction the advisor gets reliably right (4 of 4 cells) is unrepresentable in its own accounting, and the
hit rate is computed only over the boundary candidates where it is chance.

**What it did contribute:** precision in detection, the *direction* on starved reductions (4/4, even when its
number was second-best), and early naming of the legality walls. It is a **defect detector with a broken cost
model** — which is what an objective with no latency term should be.

→ [`ADVISOR-VALUE`](ADVCHAL-V2-ADVISOR-VALUE.md)

### 3.15 Which starved classes are real — one of my own flags was wrong

`rms_norm` is the only starved class the advisor wants widened. Of the two others I flagged:

| op on ≤2 cores | sum µs | max share | advisor says | real defect? |
|---|---|---|---|---|
| `rms_norm` | 1,030 | 9.48 % | widen | **✅ 6–13 %/layer every time measured** |
| `nlp_create_qkv_heads_decode` | 283 | 9.27 % | keep on 1 core | **❌ not a defect** |
| `concatenate_heads` | 154 | 7.79 % | move to DRAM | **✅ but it's the wrong *op* — §3.16** |

**`nlp_create_qkv_heads_decode` was my error, corrected before publication.** The op height-shards over batch,
so its core count *is* the batch size — perfectly, across all 23 rows (batch 1 → 1 core; batch 32 → 32 cores).
One core at batch 1 is the op's semantics, and the advisor advising 1 is **correct**.

So the "starved op" hypothesis narrows to exactly two things: the low-core reduction, and one wrong-op call.

### 3.16 A defect class the stage cannot see: the wrong op

Chasing the largest starved op in the corpus led outside the stage's question entirely.

gemma-4-12B spends **102.6 µs — 7.79 % of its full-attention window — concatenating heads on ONE core**
(verified: 24 of 24 profile instances on 1 core; the layer norms on that same core cost 9.2 µs). Every other
cell does the same logical step for **3.4 µs mean** on 16–32 cores:

| op | cells | mean µs | cores |
|---|---|---|---|
| **`concatenate_heads`** | **gemma-4-12B only** | **76.9** | **1** |
| `nlp_concat_heads_decode` | 13 others | **3.4** | 16 / 24 / 32 |

**It calls a different TTNN op.** Estimated ≈**2.4–2.6 ms/model** — **3.9× what that cell shipped** — in the
corpus's *most thoroughly screened* cell (28 measurements). I tried three in-place fixes; all hit kernel walls
(`bad optional access` ×2, then `TT_FATAL: Input tensor must be sharded`, which chains into the
`Sharded output not supported for GQA` wall two other cells already recorded).

**The stage's question is about layouts** — *which conversions does the plan not place*. This is about **op
selection**, so nothing in it could reach this: the advisor advised DRAM, the reconciliation filed it under
DRAM-advice, and the cliff check filters it out precisely because the advisor does *not* want it widened.

**The lens that found it is a cross-model comparison the stage never makes** — for the same logical operation
at the same batch, which cell is anomalously slow? It costs nothing and answers a question no single cell can
ask. → [`ADVISOR-VALUE`](ADVCHAL-V2-ADVISOR-VALUE.md) §8, [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E19

### 3.17 The largest apparent opportunity is a metric artefact — and it retracts one of my own recommendations

Generalising the cliff check to *any* op the advisor wants widened, ranked by `us × (1 − shipped/advised)`:

| op | rows | "parallelisable" µs | cells |
|---|---|---|---|
| **`linear`** | 67 | **4,970** | 13 |
| `rms_norm` | 53 | 1,057 | 9 |

`linear` dominates at 4.7× the norm, always **shipped 12 cores, advised 55–99**. Measured on north-mini, one
knob from the shipped default:

| | median ms | vs default |
|---|---|---|
| 12-core DRAM-sharded matmul (shipped) | 0.172532 / 0.172563 | — |
| wide L1-sharded — **the advisor's direction** | 0.284863 / 0.285367 | **+65.2 %** |
| MoE expert matmuls at 48/64 and 16/16 cores | 0.522919 / 0.528120 | +0.96 % / +1.96 % |

**DS matmuls are DRAM-bandwidth-bound, so core count isn't the limiting resource** — the metric is
reduction-shaped and the op is bandwidth-shaped. Corpus plus these probes: **1 win in 7 measured
matmul-widening candidates.**

⚠ **This retracts part of my action point C5.** I had recommended treating a `ds_family` grid mismatch as
screenable; that would have spent device time on a 65 % regression. The field is still worth recording; the
recommendation to screen those rows is withdrawn.

### 3.18 The biggest cost in the corpus is a layout crossing, and it is correctly invisible

Chasing §3.17 exposed a bucket my own per-op dataset had been silently dropping — `boundary`. Recovered from
the reconciliations' `disagreements[]`, corpus-wide it is **76.5 % `retilize`** (4,114 µs of 5,376), the tile ↔
row-major crossing, and the most expensive conversion class in the corpus (6.7–10.0 µs each vs 1.4–1.9 for an
L1 regrid).

| cell | kind | retilize/layer | share of layer | layers | **per model** | advisor ceiling |
|---|---|---|---|---|---|---|
| **qwen B** | **linear_attention** | **3,983.5 µs** | **25.2 %** | 48 | **191,210 µs** | **0.000 µs** |
| phi FN | dense | 63.7 | 8.8 % | 32 | 2,038 | 71.637 |
| qwen FN | full_attention | 21.2 | 2.1 % | 16 | 339 | 34.282 |

**191 ms/model = 24.4 % of qwen B's decode time — 14× every shipped win in the corpus combined.** The ops are
`UntilizeWithUnpadding` ×3 at **819 µs each** and `TilizeWithValPadding` ×2 at **671 µs each**, all on the
`add → rms_norm` edge, all **already on 109 of 110 cores**. So it is not under-parallelisation; it is the
crossing itself, and the decoder's own source says why:

> *Conv, reshape, and recurrent composite kernels currently require interleaved tensors; cross that boundary
> once after the packed projection instead of four times before four independent matmuls.*

The decoder has already minimised the *number* of crossings. **And the advisor's ceiling of 0.000 µs is
correct** — the advice places these conversions too, because they are legally required. The stage filed the cost
under `boundary`: reported, out of scope, uncredited. Honest, and enormous.

→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E20–E21

### 3.19 …and it is a shape choice, fixable in the decoder, that no advisor could have found

I first called §3.18's fix "tt-metal: accept tiled input". Reading the chain and the shapes gives a better
answer.

The conv is a causal depthwise convolution written as **permute → slice `[..., 1:]` → concat on the last dim →
multiply + sum → permute back**. From the model's own config:

| | |
|---|---|
| `linear_conv_kernel_dim` | **4** |
| conv state shape | **(1, 1, 10240, 4)** — the conv window is the **last** dim |
| tile geometry 32 × 32 → the last dim | **4 padded to 32 = 8× inflation** |
| real data / tiled+padded | **80 KB / 640 KB** |

| op | measured | effective bandwidth |
|---|---|---|
| `UntilizeWithUnpadding` | **819.4 µs** | **0.90 GB/s** |
| `TilizeWithValPadding` | **671.1 µs** | **1.10 GB/s** |

This machine's measured DRAM roofline is **~90 GB/s**. These run at **~1 %** of it — 819 µs to move 80 KB, on
109 cores. **So it is not an inherent cost and not a kernel-capability gap: it is a pathological shape.**

**Three ways to write the chain differently**, cheapest first: keep the conv window on a *leading* axis (dims
0–1 are not tile-constrained) so the permutes vanish; or replace shift-and-concat with a **circular buffer**
(overwrite slot `t % 4`, rotated weights — the layout never changes, so there is nothing to convert); or express
the depthwise conv as a **matmul against a banded matrix**, tile-native by construction.

**Could the advisor have found it? No — four independent reasons:**

| # | reason |
|---|---|
| 1 | **Row-major candidates are never enumerated** — `rowMajorEnabled` defaults `false` and the advisor's option string never sets it |
| 2 | Even enabled, `RowMajorLayoutPropagation` starts only from **integer-typed function inputs** (*"Currently restricted to integer tensor types only"*) — it deletes redundant RM→Tile on page tables, it does not build RM compute chains |
| 3 | **The score cannot price a tilize.** `requiresReshard` is a **boolean**; `LayoutScore` has no tilize/`isTiled`/element-type term. An **819 µs untilize and a 1.5 µs L1 regrid are the same value to it** |
| 4 | **Structurally, the advisor assigns layouts to a fixed graph** — it cannot delete a `permute`, `slice` or `concat`. Every fix above is a graph rewrite |

Reason 4 is load-bearing: 1–3 are fixable, but even a perfect layout assigner would miss this, because the
defect is **which axis the data lives on**, not where the tensor is placed.

→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E22

---

## 4. What makes a model advisor-compatible

Ranked by how much it actually decided outcomes.

| # | factor | why it matters |
|---|---|---|
| 1 | **The tracer can see the layer** | If the advisor can't read it, nothing recovers it. §3.5 |
| 2 | **A reduction sits on ≤2 cores** | The only class that paid double digits. §3.2 |
| 3 | **The op's neighbours accept sharded I/O** | Most common reason a legal candidate never gets a number |
| 4 | **The grid is legal at all** | Both a shard-validation rule and per-model grid helpers restrict the ladder hard. §3.4 |
| 5 | **Headroom above the noise floor** | Decided two cells outright: phi exp17 and qwen FN |

The recurring hard walls, verbatim: `Sharded output not supported for GQA`;
`nlp_concat_heads_decode requires a sharded input`; `Cos must be sharded in decode mode`; paged GQA SDPA
rejects sharded output; `shard_padded_w … trailing pad must be less than one shard width`.

---

## 5. Were the winners just worse to start with?

**For gemma-4-26B, yes — and I measured the mechanism.** Both arms' full norm ladders, same model, same
tool, same host:

| cores | 1 | 2 | 4 | 8 | 11 | 22 | 44 | 88 |
|---|---|---|---|---|---|---|---|---|
| **onA**, incumbent at **1** core | **1.8235** | — | — | — | 1.5736 | 1.5798 | **1.5750** | 1.5875 *(shipped)* |
| **FN**, incumbent at **8** cores | can't run | 1.3647 | 1.3308 | **1.3183** | 1.3179 | **1.3163** | 1.3166 | 1.3245 |

**The curve is flat from ~8 to ~44 cores. Essentially the whole win is the first step off 1 core.** onA was
on 1 and gained 13.7 %; FN's stage-02 arm had already moved it to 8, and FN has nothing left that clears its
noise floor.

⚠ **A correction to what I published earlier.** I had framed this as "the same candidate won on the slow arm
and regressed on the fast arm". The conclusion holds, but the two arms never ran the same experiment: FN's
frozen incumbent already had the norm on 8 cores, so its measurement was **8 → 88**, not 1 → 88. Worse, the
*same env knob* defaults to 88 in one arm and 8 in the other — and nothing in the stage records the
incumbent's own grid for the op under test, which is the one field that would have made this visible.

**For phi-3.5, no.** Its four arms span 1.68× in control speed and the ordering is *inverted*: the
**fastest** arm took the **largest** win (−8.75 %), and the **slowest** arm — with the **largest** ceiling
(83.6 µs/layer) — shipped **zero**.

**For the llamas, neither.** They were *already correctly placed*: I swept llama-8B's entire achievable
ladder and nothing beats the default.

**So:** the advisor is a **defect finder** more than an optimiser, and in this corpus the defect is almost
always "a reduction never got sharded". Its value is therefore **non-additive with upstream work** — fix the
placement in `$optimize` and the advisor's contribution on that model drops to near zero, which is exactly
what gemma-4-26B FN demonstrates.

⚠ One caveat on all cross-arm reading: `B` and `onA` are nominally the same arm and their controls differ by
**45 %** — larger than any advisor contribution measured anywhere in this corpus.

## 6. What we tested on hardware

Four experiments, 2026-08-03 16:07–16:23 UTC, in isolated worktrees using each cell's own harness.

| # | question | result |
|---|---|---|
| E1 | Was phi FN's discarded −13.24 % real and shippable? | **Confirmed** — reproduced to 0.02 %, and it's *more accurate* than what shipped |
| E2 | Should phi have tested the 32-core exactly-dividing grid? | **My hypothesis refuted** — 11→48 is a plateau |
| E3 | Is llama-8B's zero real? | **Confirmed** — whole ladder measured; also found the 60× cross-process floor effect |
| E4 | Was north-mini's sweep exhausted? | **No** — 16 cores is ~1 pp better; but 44/88 are *illegal*, refuting my other suggestion |
| E5 | Did gemma-4-26B onA ship the best grid? | **No** — 44 beats the shipped 88 by 12.3 µs/layer on both kinds, at PCC 1.0 |
| E6 | Why did the same candidate regress on gemma FN? | Its incumbent **already had the norm on 8 cores** — the comparison was 8→88, not 1→88. §5 corrected |
| E7 | Can a static check predict which cells have a win? | **Yes** — and the cell it flagged has a **−12.4 %/layer** win it built and never screened |
| E8 | Does tightening the harness rescue an overlapping candidate? | **No — my own proposal, refuted.** It didn't separate, and made the floor 3–4× worse |

One conclusion across E2/E4/E5/E6: **the grid a cell ships is the grid it was told, not the grid that is
fastest — and the whole win is usually just the first step off one core.**

→ [`EXPERIMENTS`](ADVCHAL-V2-EXPERIMENTS.md)

---

## 7. The stage itself

**v2 fixed measurement and broke shipping.**

| v2 got right | evidence |
|---|---|
| One fixed harness protocol | v1's floors spanned 45× by protocol alone; v2's 149 measurements are uniform. gemma-12B's floor went 18.284 → 0.712 µs and it shipped |
| Non-overlap at fixed n=5 | 3 cells declined a better-median candidate on the rule |
| `feasibility.verdict` | stops device time on unmeasurable advice; in v1 a cell shipped a win at 0.65× its floor |
| Full-model estimate **with a band** | qwen FN headlined that its gain sits inside its band |
| "The advice is a whole-graph plan" | made 2 cells extend across a blocking neighbour instead of giving up |

| v2 still gets wrong | cost |
|---|---|
| Oracle rules are contradictory and jointly force a veto | **−8.5 pp** on phi FN |
| `reconcile.py` never fills the verdicts the gate demands | 4 cells, 4 different violations; tagging records the workaround, not quality |
| Grid rule says sweep *above* the advice | **−1 pp** on north-mini; optimum was below |
| Advised core count presented as meaningful | anchors every sweep on a number with no latency content |
| Ceiling prices in-chain re-grids at 0 | a 0.000 µs ceiling sat next to a −13 % win |
| Products mandated only *across* layer kinds | only 2 of 15 cells built a within-kind product; both gained |
| Nothing checks control-plus-one-knob | the one cell that checked, failed, and remeasured everything |
| Floor treated as within-process | 60× cross-process effect unmodelled |
| Profiler vs timing protocol | 3 cells, 3 undocumented workarounds |
| Soft name/position pairing invents boundaries | 2 cells caught it against the IR |

→ [`STAGE-ANALYSIS`](ADVCHAL-V2-STAGE-ANALYSIS.md)

---

## 8. Still on the table

**From the advisor: essentially nothing new.** After matmuls are ruled out (§3.17) and
`nlp_create_qkv_heads_decode` is shown to be a batch artefact (§3.15), the actionable `chain` pool is 5,067 µs
corpus-wide: `rms_norm` 26.6 % (the proven class), then `multiply` 9.0 %, `add` 4.8 %, `slice_static` 3.1 %,
`rotary_embedding` 2.7 %, `concat` 2.0 % — all small.

**Outside the advisor: a great deal, and bigger.** Ranked by scale:

| # | opportunity | scale | kind of fix |
|---|---|---|---|
| 1 | **`retilize` on qwen's conv chain** | **191 ms/model — 24.4 %** | **decoder**: get the 4-element conv window off the 32-wide tile axis (circular buffer) — §3.19 |
| 2 | qwen's untraced linear attention | **97 %** of its decode time never advised on | tt-metal: tracer support for mutable-state `ttnn.copy` |
| 3 | `ttnn.sparse_matmul` tracer support | unblocks north-mini onA; 58–65 % of every gemma-4-26B window | tt-metal |
| 4 | `concatenate_heads` wrong-op in gemma-4-12B | ≈2.6 ms/model, 3.9× what that cell shipped | tt-metal (sharded GQA SDPA output) then a decoder change |
| 5 | phi FN's discarded combined candidate | **+8.5 pp** on that cell | stage: absolute oracle at the model's own bar |
| 6 | gemma-4-26B B's `RESIDUAL_SHARD_CORES=22`, sliding only | **−3,918 µs/model**, 26× what it shipped, *and more accurate* | ship it |
| 7 | gemma-4-26B onA at 44 cores instead of 88 | −375 µs/model, bit-identical output | ship it |
| 8 | north-mini FN at 16 cores instead of 32 | −264 µs/model | ship it |
| 9 | sweep the legal ladder both sides of the advice | 1–5 pp per affected cell | stage |

What to change in the stage and the advisor: [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).

## 9. Corrections to earlier published versions of these documents

| claim | correction |
|---|---|
| phi FN "screened 11 cores only, measured slower" | It swept 11/12/24, all faster. Rejected by the oracle, not by timing |
| phi FN missed the 32-core exactly-dividing grid | Measured: 11→48 is a plateau. Nothing was missed |
| phi FN chose its 0.999999 bar in a vacuum | The skill instructs "moves PCC at all → reject" and recommends a differential oracle |
| north-mini should have tried 44 and 88 cores | Both are illegal (`TT_FATAL`). But **16** was available and is better |
| Arm labels (`FN`/`B` inverted) | `FN` = fuse-noadvise, `B` = nofuse-noadvise, from the driver claim lines |
| gemma-4-26B: "the fusing arm had already fixed it" | The *fastest* arm is a `nofuse` arm. The variable is stage-02 quality |
| The advisor has a "fewer-cores bias" | Its ordering prefers *more* cores, at level 6 of 7. The low values come from elsewhere — §3.3, open question |
| qwen's unreachable linear layers are "~91 %" of its model time | **97 %** — recomputed from its own per-kind medians and layer counts |
| "Re-measure an overlapping candidate at 4× replays" (my proposal) | **Refuted by experiment.** No separation, and the floor got 3–4× worse (§3.7, E8) |
| Implicit: that the wins generalise across batch | They do **not** — phi is batch-32-pinned by construction (E17), and nothing records it |
| Implicit: that DS-matmul advice never wins | One *did* (gemma-4-12B `linear` 12→55c, kept), and 65 % of matmul cost was never screenable anyway (§3.12) |
