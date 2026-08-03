# advchal-v2 — read this one file

**What was measured.** Stage 02b (`$advisor-challenger`) ran on 15 decoder cells to answer one question:
how much of a decoder's speed can the shard advisor be credited with, on a decoder already optimised
without it?

**The short answer.** The advisor's value is the size of the *placement defect it happens to find* — mostly
one class of defect, a reduction stuck on too few cores. Where that defect exists it is worth 6–13 % per
layer. Where it doesn't, the honest answer is zero, and 7 of 15 cells returned one.

**Three things the stage got wrong, all measured on hardware afterwards (6 experiments):** it **discarded its own largest
win** on a contradictory correctness rule (−8.5 pp), and in **both** cells where the grid ladder had more
than one legal rung it **shipped the wrong rung** (−264 and −375 µs/model). Total left on the table:
**≈ 4.1 ms/model across three cells.**

---

## Where everything lives

| file | what's in it |
|---|---|
| **this file** | the account, and pointers |
| [`ADVCHAL-V2-IMPROVEMENTS.md`](ADVCHAL-V2-IMPROVEMENTS.md) | what to change — ideas, then action points |
| [`ADVCHAL-V2-EXPERIMENTS.md`](ADVCHAL-V2-EXPERIMENTS.md) | 6 experiments run on hardware to test the analysis |
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

## 3. The seven findings that matter

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
so ~91 % of its model time was never advised on. **This is a `$shard-advise`/tt-mlir coverage problem, not
a placement problem.**

### 3.6 A ~0 µs ceiling is not a stopping condition

The ceiling prices *boundary conversions the advice doesn't place*. A re-grid of an op that stays inside its
chain removes no boundary, so it prices at **0.000 µs**. gemma-4-26B onA recorded a 0.000 µs ceiling on both
layer kinds, screened the candidate anyway, and shipped **−12.98 %**. Two other cells trusted a similar
ceiling and shipped zeros.

### 3.7 Part of the noise floor is between processes, not within them

The first harness process of a session recorded a floor of **11.838 µs**; the identical configuration later
recorded **0.196 µs** — **60×**, from JIT-cache warmth *across* processes. Per-process warm-up cannot remove
it, and the stage mandates one process per configuration. A cell whose control ran first carries an inflated
floor, which directly changes its `feasibility.verdict`.

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

| # | opportunity | value |
|---|---|---|
| 1 | Ship phi FN's combined candidate under an absolute oracle at the model's own bar | **+8.5 pp** on that cell |
| 2 | Ship north-mini's 16-core MoE norm | **−264 µs/model**, ≈ +1 pp |
| 2b | Ship gemma-4-26B onA's 44-core norm instead of 88 | **−375 µs/model**, at PCC 1.0 |
| 3 | Tracer support for qwen's linear-attention `ttnn.copy` boundary | ~91 % of qwen's model time, currently unadvised |
| 4 | `ttnn.sparse_matmul` tracer support | unblocks north-mini onA; 58–65 % of every gemma-4-26B window |
| 5 | Sweep the legal ladder both sides of the advice in every norm cell | 1–5 pp per affected cell |
| 6 | Re-screen qwen B's geometry off the one-row worker grid | currently a hard zero |
| 7 | Re-screen phi exp17's overlapping candidate at higher replay count | its 83.6 µs/layer ceiling is the corpus's largest unrealised |

What to change in the stage and the advisor: [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md).

---

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
