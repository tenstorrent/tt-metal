# advchal-v2 — the findings in full

The detailed companion to [`READ-THIS`](ADVCHAL-V2-READ-THIS.md). That file is the few-minute version: the
verdict, the per-cell table, and the ledger of whose defect is whose. **This file is the detail** — 29 findings
and the method — and each finding points on to the file holding its raw data.

Section numbers here are the ones the other documents cite (`§3.11`, `§5`, `§8a`…); they did not change when
this file was split out of READ-THIS.

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

Strict attribution — freeze the control, count only the delta — is how this procedure gets a trustworthy number,
and it is deliberately conservative. It undercounts in at least three known ways: the ceiling prices in-chain re-grids at **0.000 µs** (§3.6), the
accounting records the one direction the advisor reliably gets right as `kept: 0` (§3.14), and nothing ever
applies the advised plan as written (§3.27). Read a cell's delta as *what this procedure credited*, and the
`reachable` figures as *what was there* — [`READ-THIS`](ADVCHAL-V2-READ-THIS.md) §2 gives both side by side.

---


---

## 3. The twenty-nine findings that matter

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
at ≈1.0. Cost: **−3,466 µs/model available, −1,267 µs shipped** — and that ceiling rose to **−4,609 µs**
once the rope was implemented as the advisor actually advised it rather than as the cell shipped it (§3.27, §8a).

**And the rule is not applied consistently, because the oracle's *construction* decides the outcome:**

| cell | same class of change | how much it moved a differential PCC | oracle built | outcome |
|---|---|---|---|---|
| gemma-4-26B onA | 1 → 88 cores | **0.0177** | absolute, vs HuggingFace, bar 0.995 | **shipped** (−12.98 %) |
| phi-3.5 FN | 1 → 11 cores | **0.0000089** | differential, vs frozen incumbent, bar 0.999999 | **rejected** |

The cell whose change perturbed the output ~2,000× more shipped it. *(Not a controlled comparison — gemma's
figure is on synthetic weights, phi's on real, different models. The asymmetry in outcome doesn't depend on
that.)*

**And a differential oracle cannot tell you which side moved.** I built the absolute oracle action **A1** prescribes —
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

### 3.11 The stage credited the advisor with 64 % of what it actually found — at most

Comparing what each cell shipped against the best configuration reachable from the advisor's *own* directions
on the *same* decoder (my re-measurements included):

| cell / kind | incumbent | shipped | best measured | stage says | best says |
|---|---|---|---|---|---|
| phi FN dense | 0.808757 | 0.769096 | **0.663507** ¹ | −4.90 % | **−17.84 %** ¹ |
| g26 B sliding | 1.258327 | 1.254000 | **1.101768** | −0.34 % | **−12.44 %** |
| g26 onA sliding | 1.823508 | 1.587511 | **1.574985** | −12.94 % | −13.63 % |
| nm FN sliding MoE | 0.577971 | 0.518022 | **0.512764** | −10.37 % | −11.28 % |
| g26 FN sliding | 1.341153 | 1.318449 | 1.316251 | −1.69 % | −1.86 % |
| llama-8B dense | 0.665237 | = incumbent | = incumbent | — | **tested**: whole ladder swept, nothing beats the default (E3) |
| phi A dense, phi B dense | 0.656989 / 0.788610 | = shipped | *unknown* | −7.58 % / −5.09 % | **not tested** — never probed for a better configuration |

**Shipped 13,601 µs/model. Reachable 21,368 µs/model — 1.57×**, so the stage credited the advisor with **64 %**
of what it found.

⚠ **"Reachable" is a lower bound, and so is the 64 %.** Five cell/kinds contribute a measured better
configuration; llama-8B contributes a **tested** zero; **phi A and phi B contribute zero because I never probed
them**, not because nothing is there. The seven cells outside this table were not re-measured at all. So 64 % is
the *highest* share the stage could have credited given what I measured — the true share is lower by an unknown
amount. The missing third is not new ideas: it is the same directions at a different grid, the same
candidate past an oracle that should have passed it, or — the largest single piece — **the advisor's own plan
applied as written instead of assembled chain by chain**.

¹ phi FN's best is from the 2026-08-07 re-measurement (control 0.807535 → 0.663507, §3.27), which is the advised
rope plus the advised 11-core norm. The earlier 0.700431 used the *interleaved* rope the cell shipped. Its
control run differs from this table's 0.808757 by 1.2 µs of run-to-run drift; the Δ is computed within each run.

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

### 3.14 For this win class, the advisor was not necessary

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
| `concatenate_heads` | 154 | 7.79 % | move to DRAM *(but see §3.28 — for `nlp_concat_heads_decode` that DRAM layout is a fallback after the advisor declared the op `unfixable`, not a recommendation)* | **✅ but it's the wrong *op* — §3.16** |

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

### 3.19 That 191 ms is a shape choice, fixable in the decoder, that no advisor could have found

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

1. **Row-major candidates are never enumerated** — `rowMajorEnabled` defaults `false` and the advisor's option string never sets it
2. Even enabled, `RowMajorLayoutPropagation` starts only from **integer-typed function inputs** (*"Currently restricted to integer tensor types only"*) — it deletes redundant RM→Tile on page tables, it does not build RM compute chains
3. **The score cannot price a tilize.** `requiresReshard` is a **boolean**; `LayoutScore` has no tilize/`isTiled`/element-type term. An **819 µs untilize and a 1.5 µs L1 regrid are the same value to it**
4. **Structurally, the advisor assigns layouts to a fixed graph** — it cannot delete a `permute`, `slice` or `concat`. Every fix above is a graph rewrite

Reason 4 is load-bearing: 1–3 are fixable, but even a perfect layout assigner would miss this, because the
defect is **which axis the data lives on**, not where the tensor is placed.

→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E22

### 3.20 The stage already measures the best predictor of where wins are — and reads 3 columns out of it

`tt-perf-report` emits, per op: **`Total %`**, `Bound`, `Cores`, `DRAM %`, `FLOPs %`, math fidelity — plus a
**stacked report with an `Op Category` column (`Compute` / `TM` / `DM` / `Other`)**, a roofline summary, and
per-op advice like *"Increase grid size (currently using 2.0)"*.

**`reconcile.py` reads three columns**: `OP CODE`, `DEVICE KERNEL DURATION [ns]`, `CORE COUNT`. The skill and
the gate never mention `--group-by category`, `--summary-file`, `--stacked-csv`, the roofline or the advice.
14/14 cells saved the CSV; only 4–5/14 saved the summary, stacked report or terminal output.

Scoring every cell's bounded window with the tool's own classifier:

| cell | Compute % | **TM %** | DM % | Other % |
|---|---|---|---|---|
| **qwen B** | 42.6 | **53.3** | 1.5 | 2.6 |
| phi A / B / FN / exp17 | 38.6–54.5 | **17.1–26.3** | 1.9–11.4 | 13.3–36.2 |
| g26 B, gemma-12B, qwen FN | 75.0–78.5 | 4.3–8.6 | 2.4–4.5 | 8.4–14.7 |
| **llama-1B / llama-8B** | **80.7 / 81.4** | **1.1 / 0.9** | 8.6 / 5.4 | 9.6 / 12.3 |
| **mean** | **63.2** | **16.5** | **5.6** | **14.7** |

**This rubric predicts the corpus's outcomes better than the advisor does.** The two cells that returned
zeros after exhaustive screening are exactly the two cleanest (TM 0.9 %, 1.1 %); the cells where wins
were found or missed are the dirty ones (phi 17–26 %, qwen B 53 %).

**And splitting the non-compute time answers "could the advisor help?" directly:**

| | mean share of window | can a layout assigner remove it? |
|---|---|---|
| **layout-induced** (tilize/untilize/typecast/fill-pad/interleaved↔sharded/reshard/copy) | **6.0 pp** | **potentially — with a conversion cost model (D0). Not today: `requiresReshard` is a boolean** |
| **graph-structural** (permute/transpose/reshape/slice/concat/head-ops) | **12.5 pp** | **no — only a graph rewrite** |

So **≈1/3 advisor-reachable, 2/3 not.** qwen's 191 ms `retilize` is in the second bucket, which is why §3.19
concluded it needs a chain rewrite rather than a placement.

**The 4-line category fix, measured:** applying it collapses the `Other` bucket from **12.8 % mean to 0.2 %**,
with 11 of 13 cells reaching exactly 0.0 — 8.1 pp into Compute, 4.6 pp into TM. The cell ordering is unchanged,
so the rubric was already usable; the fix makes the absolute numbers meaningful.

→ [`PERF-REPORT-AUDIT`](ADVCHAL-V2-PERF-REPORT-AUDIT.md)

### 3.21 The advisor's option space, swept — and my row-major recommendation is refuted

Ran `ttnn-advise mlir` directly (no device needed) on phi's own IR, one option at a time:

| run | ops | reshards | **row-major layouts** | outcome |
|---|---|---|---|---|
| baseline ×2 | 35 | 39 | **0** | **identical — the advisor is deterministic** |
| `opt-level 3` | — | — | — | **FAILS.** `TTNNPipelines.h:592` validates 0..2 |
| `disable-dram-sharded-matmul=true` | 35 | 39 | **0** | identical to baseline |
| `row-major-enabled=true` | 35 | 38 | **0** | 4 matmuls narrowed `width/1x96` → `block/1x11` |

**Zero row-major layouts, with the flag on.** They *are* enumerated either way, and the flag lets far more
through — the pipeline log grows **3.3 MB → 35 MB (10.8×)** — but every one is rejected:

```
TT_FATAL: Input tensor layout must be TILE but got Layout::ROW_MAJOR
```

**The blocker is that TTNN ops reject row-major input** — tt-metal territory, out of scope. So my D0b
recommendation ("enumerate row-major") is withdrawn, and §3.19's reason 4 gets stronger: the advisor can neither
rewrite the graph nor legally place anything in row-major. Both halves of the `retilize` problem are outside its
reach.

What survives: **price conversions as a cost rather than a boolean** — the only advisor-side lever, operating
entirely within TILE layouts, which is the whole space it legally has.

→ [`COUNTERFACTUALS`](ADVCHAL-V2-COUNTERFACTUALS.md) §E23

### 3.22 What a shipped win actually is, op by op — movement, and a buffer type

The one win with a clean before/after profile pair (phi FN's `rope_l1`), cross-checked on phi B:

| | phi FN | phi B |
|---|---|---|
| profiled window | 725.2 → 654.0 µs (**−9.82 %**) | 699.8 → 624.0 µs (**−10.82 %**) |
| **Compute** | 434.1 → 430.6 (**−3.5**) | 462.3 → 456.0 (**−6.4**) |
| **TM** (tensor manipulation) | 212.1 → **146.0** (**−66.2**) | 224.4 → **154.9** (**−69.5**) |
| DRAM roofline | **15.9 % (81 GB/s) → 17.6 % (90 GB/s)** | — |

**93 % / 92 % of the win is movement reduction; compute is flat.** By op class, near-identical across the arms:
`Permute` −38.95/−38.56 µs (**55 % / 51 %** of the change), `Concat` −14.02/−14.21 (20/19 %), `Slice`
−8.44/−8.40 (12/11 %). Six `Permute` ops disappear entirely; `Concat` on 110 cores halves.

**And it is not a sharding change.** By `Input 0 Memory`:

| | BEFORE | AFTER |
|---|---|---|
| `DRAM_INTERLEAVED` | 587.2 µs, 81.0 %, **49 ops** | 433.7 µs, 66.3 %, **21 ops** |
| `L1_INTERLEAVED` | — | **83.6 µs, 22 ops** (new bucket) |
| `L1_WIDTH_SHARDED` / `L1_HEIGHT_SHARDED` | 79.9 / 58.0 µs | 79.9 / 56.7 — **unchanged** |

**28 ops moved DRAM → L1 interleaved and no shard spec changed** (`DRAM Sharded` is `False` on every row of both
files). A win from a *shard* advisor is a change of buffer type — consistent with `LayoutScore` ranking `isL1`
**first** (§3.3).

**Does the shipped result follow the advice?** Of the 27 ops the stage bucketed as `chain`: **4 follow it, 6 took
the buffer type only** (L1 but not the advised sharding — which runs, and is 2.1× better, E25), **9 do not**, 1 matches the layout family but not the
grid, and **7 are undecidable** because the op↔advice pairing is positional. And of the 2 ops the stage labels
`agrees_with_shipped` — its term for *"we already do what the advice says, nothing to screen"* — **one does not
actually agree**: `typecast`, advised `l1/height_sharded/1x1`, shipped 1 core DRAM-interleaved.

**§3.24 gives the reason for every one of them**, and the reasons concentrate hard: nine of the twelve "no" ops
are one rejected oracle call.

→ [`PHI-BEFORE-ADVISED-AFTER`](ADVCHAL-V2-PHI-BEFORE-ADVISED-AFTER.md)

### 3.23 ~~The advisor validates plans the runtime refuses to run~~ — **RETRACTED. The advice was legal, and the fastest thing measured**

**This finding was wrong and the error was mine.** I reported that the advised `l1/height_sharded/32x1` for
phi's RoPE body could not run, on two `TT_FATAL`s, and escalated it to a tt-mlir ↔ tt-metal validation gap.
Withdrawn in full.

**What the advisor actually specified**, from `final_ir.mlir` (the authoritative artefact, which I had not read):

```
#ttnn_layout24 = <32x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>,
  core_ranges = <[core_range<(0,0),(10,1)>, core_range<(0,2),(9,2)>]>
```

Shard shape **(32, 64)** — two full tiles, padded and tile-aligned — on **32** cores (two ranges, 22 + 10).
My probe used shard **(32, 48)**, the logical width, which the advisor never specified; and I sharded the
`concat` output while leaving its inputs interleaved, where the IR shards the inputs first.

**Isolated single-op test, the advisor's exact config, one op at a time on device:**

| op | config | result |
|---|---|---|
| `ttnn.neg` (1,32,32,48) | shard (32,64), 32 cores | **OK** |
| `ttnn.concat` 2×(…,48)→(…,96) | shard (32,96), 32 cores | **OK** — the case I claimed impossible |
| `ttnn.multiply`, `ttnn.add` (…,96) | shard (32,96), 32 cores | **OK** |
| `ttnn.slice` → (…,48) | shard (32,64), 32 cores | **OK** |
| *control:* `ttnn.neg` shard **(32,48)** — my old probe | | **FAIL** `tensor_layout.cpp:162` |

**And implemented verbatim it is the best form measured** (E25, the cell's own harness, fresh process each):

| form | median ms | Δ | differential PCC |
|---|---|---|---|
| frozen incumbent | 0.807535 | — | — |
| **what phi FN shipped** (L1 interleaved) | 0.768104 | −4.88 % | 1.0 |
| what phi B shipped (sharded multiply/add) | 0.751277 | −6.97 % | 1.0 |
| **the advisor's IR, verbatim** | **0.723320** | **−10.43 %** | **1.0** |

Strict non-overlap between all three; all bit-identical to the incumbent. **The stage's unproven deviation cost
5.55 pp ≈ 1.43 ms/model.** There is no validation gap, no illegal advice, and no tt-metal bug.

→ [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §4–§5

---

### 3.23b The advised core count was misreported to every cell — by the stage, from a lossy field

`report.json` prints `l1/height_sharded/32x1 cores=(0,0)-(10,1)`; `reconcile.py:194` parses the `cores=` range
and reports **22**. The IR's `CoreRangeSet` has **two** ranges — 22 + 10 = **32**. The grid string `32x1` was
right all along; `cores=` is a lossy single-range rendering.

Validated against three decision traces (`beam[0].score.coreCount`, the value `LayoutScore` compares):
grid-string product correct **22/22, 10/10, 17/17**; bounding box correct 2/22, 1/10, 2/17. Applied to all 15
cells: **476 of 816 advised ops (58.3 %) carry an understated core count** — 22→32 (76 ops), 88→90 (230),
77→80 (50), 88→96 (42), 1→32 (8).

**Consequence:** two phi cells recorded themselves as *overriding* the advisor while agreeing with it. phi B's
own `rejected_knobs` reads *"advisor core counts 11/22 alone (not recommendations; shipped chain uses exact
batch-dividing 32-core height shards)"* — **32 is exactly what the advisor advised.**

**Fault: the stage's, and it is a one-line fix** (use the grid product, not the bounding box) → C5f. The
`report.json` flattening is worth fixing too, but nothing is lost by the optimizer — `final_ir.mlir` carries the
full range set.

→ [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §1

---

### 3.24 Why each op did or didn't follow the advice — and it is mostly one decision

I put a sourced reason against all 56 ops of phi FN's window. The distribution:

| follows the advice? | ops | dominant reason |
|---|---|---|
| — no advice exists (`boundary` + `untraced`) | 25 | conversions live in `reshards[]`, not `ops[]` |
| **no** | 12 | **the oracle veto (9 of 12)** |
| **undecidable** | 7 | the op↔advice pairing is a positional guess |
| **buffer only** | 6 | the advised sharding was never tried — it runs, and is faster (§3.23, E25) |
| **yes** | 4 | `l1/interleaved`, which specifies no grid |
| **family only** | 2 | right space and layout family, wrong grid |

**One oracle call explains twelve of the twenty-seven actionable ops.** The reconciliation groups disagreements
into *chains* and measures the chain, not the op. Two chains — `dense:0` (`rms_norm`, `linear`,
`nlp_create_qkv_heads_decode`, `embedding` ×2, **178.4 µs**) and `dense:11` (`linear` ×2, `add`, `rms_norm`,
`slice_static` ×2, `multiply`, **196.1 µs**) — were screened together as `advisor_norm_cores=11`, **measured
0.745905 ms against a 0.807152 ms incumbent (−7.6 %), and 0.700267 ms combined with the rope win (−13.3 %)**.
Both are marked `verdict: rejected`. Not on time — on `oracle_pcc_bar: 0.999999` versus a measured
`0.9999910667`. This is §3.4 seen from the op side: **the 12 unimproved ops are not a fact about the advice, they
are a fact about one threshold.**

**What `chain` means in the stage's own label.** `reconcile.py` assigns each profiled op one of five buckets,
first match wins: `boundary` (a movement op — no advice exists), `untraced` (absent from the advisor's graph),
`dram_resident` (advisor says DRAM, shipped is sharded), `agrees_with_shipped` (advised cores == shipped cores,
*or* both sides DRAM-sharded), and **`chain`** — everything else, i.e. **the advisor wants it in L1 and the
advised core count differs from the shipped one**. Consecutive `chain` ops are then grouped into a maximal
L1-resident run, broken by any conversion or DRAM placement, and **that run is the unit that gets measured and
shipped**. So `chain` = *"this op is part of a screenable candidate"*. Ids are `<layer_kind>:<index>`;
`dense:b14`-style ids are boundary-derived chains — a lone conversion the advisor said not to do, promoted to a
candidate on its own.

**Two new defects in the stage's own script, both found by doing this:**

| defect | evidence |
|---|---|
| `agrees_with_shipped` **never compares the memory space** — only core count, or the DS family. So an op advised into L1 and shipped in DRAM reads as agreement. | phi FN has 2 such rows and **1 is wrong** (`typecast`: advised `l1/…/1c`, shipped `1 core DRAM`). → C5c |
| `pair_confidence: position` is recorded, documented as *"a guess"*, and then ignored downstream. **11 of 31 paired rows (35 %) are positional.** | It misled *this* analysis: 7 rows I had called findings are guesses. → C5e |

→ [`PHI-BEFORE-ADVISED-AFTER`](ADVCHAL-V2-PHI-BEFORE-ADVISED-AFTER.md) for the full per-op table with the
`why` column and its legend.

---

### 3.25 Corpus-wide: the advisor's L1 call was followed widely, its grid in 3 of 9 cells

§3.24 answered this for one cell. Across all 15, at chain level:

| chain verdict | chains | µs | share | meaning |
|---|---|---|---|---|
| `below_threshold` | 108 | 3325.9 | **60.0 %** | **dismissed without a measurement** |
| `rejected` | 55 | 1063.8 | 19.2 % | implemented, measured, lost or vetoed |
| `kept` | 58 | 589.3 | **10.6 %** | the advice shipped |
| `not_measurable` | 26 | 568.4 | 10.2 % | the cell's ceiling was under its noise floor |
| `hard_error` | 1 | 0.0 | — | implementing it hit a `TT_FATAL` |

**58 of 248 chains kept, 589 of the 5,547 µs the stage counted as disagreed-on = 10.6 %.** **17 of 26 (cell,
layer-kind) pairs kept zero.**

⚠ **That denominator is inflated by a third.** With advised core counts corrected (§3.23b), **59 of 334 `chain`
rows stop being disagreements at all — 1,908 µs, 34.4 %** — because the understated count pushes agreeing rows
into `chain`. Against the genuine 3,639 µs the followed share is **16.2 %**. And `dram_resident` cannot just be
added to the denominator: 41 of 54 advisor-declared-`unfixable` rows sit in it (§3.28). The `kept` figure is also
an *upper bound*, since it means the chain shipped, not that the advised geometry was implemented.

**Buffer type vs geometry, over the 9 cells that changed anything:**

| what shipped | cells |
|---|---|
| the advised sharding strategy **and** core count | **3** — gemma-4-26B onA (`width_sharded`, 88), phi onA (`height_sharded`, 32), phi B (32, on the 96-wide ops) |
| the advised strategy, a **self-chosen grid that measured better** | **1** — north-mini FN (32 against the advised 22) |
| buffer type / boundary only, **no grid** | **5** — gemma-4-12B, gemma-4-26B FN, gemma-4-26B B, phi FN, qwen FN |

Two of those three had recorded themselves as *overriding* the advisor while agreeing with it — phi B's
artefacts say *"advisor core counts 11/22 alone (not recommendations)"*, and 32 is what the advisor advised
(§3.23b). **The one cell that did measure the advised grid head-to-head against its own found its own better:**
north-mini FN screened the advised 22 as `advisor_moe_norm_22` and recorded it *"slower than the 32-core winner
for both MoE kinds"*. So the ranking that gets used is `isL1` (level 1 of `LayoutScore`); the one that does not
is `coreCount` (level 6 of 7) — §3.3.

**Two of the three biggest wins came from cells whose own arithmetic said not to bother.** gemma-4-26B onA and
north-mini FN both shipped a widened RMSNorm; both had layer kinds with `ceiling_us = 0`, verdict
`not_measurable`, every chain `below_threshold`, and **0 kept chains recorded**. They screened anyway and
booked −12.98 % and −9.26 % per layer. The ceiling prices boundary conversions only, so an in-chain re-grid —
exactly what a 1→88-core norm is — is worth `0.000 µs` to it. This is §3.13/D0 confirmed at corpus scale.

**And 70 of the 134 dismissed chains are ≥ 5× their own cell's noise floor**, up to 282×, including chains at
19.3 %, 14.5 % and 13.9 % of their profiled window. *(Caveat: a chain's µs is its ops' incumbent cost, not the
claimed saving — 57× the floor means a **1.7 % saving on those ops would already be measurable**, not that a
57× win exists.)*

→ [`ADVICE-FOLLOWED`](ADVCHAL-V2-ADVICE-FOLLOWED.md) for the per-cell tables and the full dismissed-chain list.

---

### 3.26 Was the exact advice tried, and tried first? In 7 of 15 cells

The principle: the advisor's advice implemented verbatim should be candidate #1, and every deviation needs a
measured reason. From each cell's own `measured_at` chronology:

| | cells |
|---|---|
| tried the exact advice, **first**, and either shipped it or measured a regression against it | **6** — gemma-4-26B onA, gemma-4-26B FN, north-mini FN, north-mini B, qwen3-27B B, phi-3.5 onA |
| tried it, but **after** the candidate it shipped | 1 — qwen3-27B FN |
| tried it **partly** | 3 — llama-3.1-8B, llama-3.2-1B, phi-3.5 B |
| **never tried it**, no reason recorded | **4** — phi-3.5 FN, phi-3.5 exp17, gemma-4-12B exp11, gemma-4-26B B |
| tried nothing at all | 1 — north-mini onA (`not_measurable`) |

**Where the protocol was followed it worked cleanly** — all 7 either shipped the advice or have a measured
regression on file. north-mini FN is the model case: it measured the advised 22 (0.5432), then its own 32
(0.5184), then 64 (0.5733), and shipped 32 with the comparison recorded.

**The 4 unproven deviations are all the same deviation:** take the advisor's L1 placement, drop its sharding.
**gemma-4-12B ran 52 measurements without ever trying an advised grid.** And in the one cell where I could
measure it, the deviation cost 5.55 pp (§3.23, E25).

→ [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) [`READ-THIS`](ADVCHAL-V2-READ-THIS.md) §2

---

### 3.27 The screening order is the biggest single defect: build-up cost phi FN 3.7×

The skill instructs build-up (`SKILL.md` §4, verbatim): *"Screen, in the order the reconciliation gives. Each
chain as one unit, one variable per measurement, against the frozen incumbent."* It never asks for the advised
plan to be applied as a whole, and never mentions `final_ir.mlir` — the only artefact carrying the complete plan
with its shard shapes.

**Measured on phi FN — its own harness, fresh process per configuration, strict non-overlap throughout:**

| configuration | median ms | Δ | differential PCC |
|---|---|---|---|
| incumbent | 0.807535 | — | — |
| **what the cell shipped** | 0.768104 | **−4.88 %** | **1.0** |
| norm 11 only (advised grid) | 0.747428 | −7.44 % | 0.99999107 |
| **rope as advised** | 0.723320 | **−10.43 %** | **1.0** |
| the cell's own best, discarded on the oracle | 0.700120 | −13.30 % | 0.99999107 |
| **rope as advised + advised 11-core norm** | **0.663507** | **−17.84 %** | 0.99999107 |

**Applying the advisor's placement is 3.7× what the cell shipped.** And note the PCC column: the advised RoPE
sharding is **bit-identical**, so **−10.43 % was available with no correctness question at all**. The oracle
(§3.1) did not cost this cell the 5.55 pp — not trying the advice did. The 0.9999911 comes wholly from the norm
re-grid: the same number with rope off, interleaved, or advised, which is reduction-order behaviour, not a
shard-spec bug.

**Corpus-wide the ordering shows up in the results.** The three best outcomes are the three cells whose *first*
candidate was the advisor's placement:

| cell | first candidate | Δ |
|---|---|---|
| gemma-4-26B onA | **the advised plan** | **−12.98 %** |
| north-mini FN | **the advised plan** | **−9.26 %** |
| phi-3.5 onA | **the advised geometry** | **−7.58 %** |
| … | | |
| phi-3.5 FN | no | −4.88 % *(−17.84 % available)* |
| qwen3-27B FN | no | −2.25 % |
| gemma-4-12B exp11 | no — 52 measurements, no advised grid | −1.83 % |
| gemma-4-26B B | no | −0.47 % |

*(Correlation — cells also differ in how much defect there was to find. What lifts it above correlation is that
phi FN's counterfactual is measured: same cell, same incumbent, same harness.)*

**Why build-up loses mechanically:** 60 % of the disagreed-on cost sits in `below_threshold` chains that are
individually unmeasurable and collectively obvious — phi FN's own norm chains were 178 µs and 196 µs. The skill
already knows this: its `aggregate_only` verdict says *"apply the top chains together as one candidate first"* —
but only as a fallback when no single chain clears the floor. **The corpus says it should be the default.**

**The change (F5):** make the advisor's complete plan, built from `final_ir.mlir`, candidate #1; if it will not
run, remove only the failing item with a single-op test naming it; then **ablate downwards** — an advised item
whose removal is *faster* is a real finding about the advisor, and today that signal cannot be generated at all
because such items are never applied. Same device time; it is a reordering.

→ [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §7–§9,
[`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) §F5

---

### 3.28 The advisor reports what it cannot place. The stage screens it anyway — 41 times

`report.json` carries an `unfixable_ops` list: ops the advisor could not place, each with the exact runtime
error, obtained by querying tt-metal's own constraint machinery. phi FN's:

```json
"unfixable_ops": [{"op": "ttnn.nlp_concat_heads_decode",
  "reason": "... TT_FATAL @ nlp_concat_heads_decode_device_operation.cpp:44: input_tensor.is_sharded()
             info: Input tensor must be sharded"}]
```

`reconcile.py:603` reads the field — but **only** to annotate the `untraced` bucket's informational note. An
unfixable op that lands in `dram_resident` or `chain` is never checked against it, and
`nlp_concat_heads_decode` lands in `dram_resident`, where the reconciliation labels it *"advisor placed it in
DRAM — that is advice, and it disagrees with a sharded shipped op."*

**Corpus-wide: 54 unfixable declarations, 41 still presented as screenable advice.** The ops are
`nlp_concat_heads_decode` (every cell), `rotary_embedding`/`rotary_embedding_llama` and `repeat`. `SKILL.md` and
the stage prompt never mention the field.

**Cells then spend device time rediscovering the advisor's own written errors.** phi FN's
`advisor_sdpa_concat_l1` knob and its `dense:b43` chain both record the same string that was already in
`unfixable_ops`; gemma-4-26B FN has the same for `sharded_sdpa_output_extension`; phi exp17 hit it
independently. **I confirmed the constraint with an isolated single-op test — a DRAM-interleaved input to
`nlp_concat_heads_decode` fails at `device_operation.cpp:44` — so the advisor's declaration was right and the
waste is entirely on the consumer side.** → C5g.

**And it revises the `dram_resident` bucket's premise:** for an unfixable op, the `dram/interleaved` in `ops[]`
is a fallback after a declared failure, not a recommendation to screen.

Separately, applying the advisor's remaining big item — the `gate_up` matmul as a DRAM-sharded matmul, exactly
as the IR specifies — is **neutral** (0.806777 vs 0.807535 incumbent; 0.664100 vs 0.663507 stacked, both inside
the floor). So the rope + norm −17.84 % is essentially the whole available win from phi FN's advised plan, and
the matmul item gets dropped *with a measurement behind the decision* — which is the ablation half of F5 working.

→ [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §10–§11

---

### 3.29 The advisor is topology-sensitive and memory-config-blind — which is why replaying its advice is valid, and re-advising is cheap

§3.27–§3.28 apply each cell's *committed* advice — captured once against the frozen incumbent — to a decoder that
had progressively been modified, which on its own would make those numbers estimates. Re-running `ttnn-advise` on
the diverged graphs settles it: **the advice is byte-identical across all four graphs**, because
the advisor discards the input's memory configs and re-places everything, so it responds to graph *topology* and
not to the memory-config changes I made. My control run also reproduces the cell's committed advice exactly, so
the advisor is deterministic at pin `618cd4e75d`. **`ttnn-advise` costs ~18 s end to end** (18.4/18.4/18.1/18.6 s
measured) — less than one harness measurement, so there is no cost argument for screening against a single
start-of-run capture. Separately: the capture **monkey-patches `_decode_rope`** with a DRAM-staging stand-in,
so the advisor never sees the cell's real RoPE. Full accounting in
[`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §12, and it is action **F6**.

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

**From the advisor, as new op classes: essentially nothing.** After matmuls are ruled out (§3.17) and
`nlp_create_qkv_heads_decode` is shown to be a batch artefact (§3.15) — and note it is *also* one of the phantom
disagreements of §3.23b, advised 32 and shipped 32 in every cell — the actionable `chain` pool is 5,067 µs
corpus-wide: `rms_norm` 26.6 % (the proven class), then `multiply` 9.0 %, `add` 4.8 %, `slice_static` 3.1 %,
`rotary_embedding` 2.7 %, `concat` 2.0 % — all small.

### 8a. Placement wins the advisor's data located — ≈9.2 ms/model

**This is the "≈9.2 ms still on the table from placement" in the opening summary, itemised.** All four were
found from the advisor's own per-op output, using the static check in §3.8. But *located by the advisor* and
*equal to what the advisor recommended* are different things, and only one of the four is both:

| # | cell | winning knob | advised value | µs/model | was the winning value the advice? |
|---|---|---|---|---|---|
| 1 | **phi-3.5 FN** | rope as advised + `norm_cores=11` | rope L1 + **11** | **−4,609** | **yes — pure advice-following** |
| 2 | gemma-4-26B B | `RESIDUAL_SHARD_CORES=22`, sliding only | **88** | −3,918 | no — the cell's own ladder value; 26× what it shipped, *and more accurate* |
| 3 | gemma-4-26B onA | `NORM_CORES=44` instead of the 88 it shipped | **88** | −375 | no — 44 beats the advised 88, bit-identically |
| 4 | north-mini FN | `moe_norm_cores=16` instead of the 32 it shipped | **22** | −264 | no — 16 beats both the advised 22 and the shipped 32 |
| | | | | **≈9,166** | **4,609 of 9,166 = 50 % is advice-following** |

**So the answer to "where did this come from?": entirely from the advisor's data, and half of it from the
advisor's actual recommendation.** The advisor's per-op output located every one of the four — no non-advisor
analysis was needed to find them. What it did not supply in three of the four is the *right grid*: it named the
op and the direction (widen this starved reduction) and a value that was beaten by one found by sweeping the
legal ladder around it. That is why item 7 of §8b exists, and why D1 (add a latency term to `LayoutScore`) is
an advisor-side action rather than a stage one.

⚠ **Item 1 grew from −3,466 to −4,609 µs/model in the 2026-08-07 pass.** The earlier figure used the rope in
the *interleaved* form the cell shipped; with the rope in the form the advisor actually advised, the same cell
gives 0.807535 → 0.663507 ms/layer × 32 layers (§3.27). The corpus total moved 8.0 → **9.2 ms/model** with it.

### 8b. Outside the advisor entirely — a great deal, and bigger

| # | opportunity | scale | kind of fix |
|---|---|---|---|
| 1 | **`retilize` on qwen's conv chain** | **191 ms/model — 24.4 %** | **decoder**: get the 4-element conv window off the 32-wide tile axis (circular buffer) — §3.19 |
| 2 | qwen's untraced linear attention | **97 %** of its decode time never advised on | tt-metal: tracer support for mutable-state `ttnn.copy` |
| 3 | `ttnn.sparse_matmul` tracer support | unblocks north-mini onA; 58–65 % of every gemma-4-26B window | tt-metal |
| 4 | `concatenate_heads` wrong-op in gemma-4-12B | ≈2.6 ms/model, 3.9× what that cell shipped | tt-metal (sharded GQA SDPA output) then a decoder change |
| 5 | phi FN's discarded combined candidate | **+8.5 pp** on that cell | stage: absolute oracle at the model's own bar |
| 6 | **screen the advisor's plan as candidate #1, then ablate** | **3.7× on the one cell measured** (§3.27) | stage: F5 |
| 7 | sweep the legal ladder both sides of the advice | 1–5 pp per affected cell — items 2–4 of §8a are exactly this | stage |

What to change in the stage and the advisor: [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).

## 9. What this analysis got wrong

Every claim I published as fact and later retracted, downgraded or re-derived — 30 of them, grouped by the
error pattern that produced them, each with the check that would have caught it first. Kept as a separate file
because the patterns are what transfer, not the individual corrections:

→ **[`ADVCHAL-V2-ANALYST-PITFALLS.md`](ADVCHAL-V2-ANALYST-PITFALLS.md)**

It also records what is **still unverified** in this corpus, so nobody inherits an open question as a fact.
