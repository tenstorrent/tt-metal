# Was the advisor's advice tried — and tried *exactly*, first?

The principle this file tests: **the advisor's advice, implemented verbatim, should be the first candidate.
Every deviation from it needs a recorded, measured reason.** For each of the 15 cells: was the exact advice
tried, was it tried first, and is every deviation proved?

> ⚠ **This file retracts a finding of mine.** I previously reported that the advisor recommended an
> *unrunnable* layout for phi's RoPE body and called it a tt-mlir/tt-metal validation gap (old §3.23, D6, E24).
> **That was wrong, and the error was entirely mine.** See §5. The advisor's advice is legal, runnable,
> bit-identical, and the fastest of everything measured.

---

## 1. First: the advised core count was misreported to every cell

Before any cell can be judged on faithfulness, the advice it was *shown* has to be established.

`report.json` renders each chosen layout like this:

```
l1/height_sharded/32x1 cores=(0,0)-(10,1)
```

`reconcile.py:194` parses the `cores=` range and reports `advised_cores = 22` (11 × 2). **That is not what the
advisor chose.** From the authoritative `final_ir.mlir` for the same op:

```
#ttnn_layout24 = <32x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>,
  core_ranges = <[#ttnn.core_range<(0,0), (10,1)>, #ttnn.core_range<(0,2), (9,2)>]>
```

**The `CoreRangeSet` has two ranges — 22 + 10 = 32 cores.** `report.json` prints only the first, so the count is
silently truncated. The `AxB` grid string (`32x1`) is right; the `cores=` field is a lossy rendering of it.

**Validated against the advisor's own decision traces** (`beam[0].score.coreCount`, the value `LayoutScore`
actually compared):

| cell | name-matched ops | grid-string product correct | `cores=` bounding box correct |
|---|---|---|---|
| phi-3.5 FN | 22 | **22 / 22** | 2 / 22 |
| phi-3.5 B | 10 | **10 / 10** | 1 / 10 |
| phi-3.5 exp17 | 17 | **17 / 17** | 2 / 17 |

Applying the validated rule to all 15 cells' `report.json`: **476 of 816 advised ops — 58.3 % — carry an
understated core count.** The largest classes:

| reported | actually | ops |
|---|---|---|
| 88 | **90** | 230 |
| 22 | **32** | 76 |
| 77 | **80** | 50 |
| 88 | **96** | 42 |
| 55 | **64** | 23 |
| 1 | **32** | 8 |

**Whose fault.** Two separate things, neither of them the optimizer's *decision*:

1. **The stage's, and it is the cheap fix:** `reconcile.py` derives `advised_cores` from the `cores=` bounding
   box when the grid string next to it is already correct. One line. No build. → new action **C5f**.
2. **A reporting loss in `report.json`:** a multi-range `CoreRangeSet` is flattened to one range. The
   information survives in `final_ir.mlir`, so nothing is lost by the *optimizer* — only by its summary. Worth
   fixing, but the consumer fix above removes the consequence.

**Consequence for the corpus.** Two phi cells recorded themselves as *overriding* the advisor when they were
agreeing with it. phi B's `rejected_knobs` says:

> *"advisor core counts 11/22 alone (not recommendations; shipped chain uses exact batch-dividing 32-core
> height shards)"*

**32 cores is exactly what the advisor advised.** The cell reached the advisor's number and filed it as its own.

---

## 2. What each cell actually tried, in order

From each measurement file's own `measured_at` timestamp — the real chronology, not the write-up.

| cell | was the exact advice tried? | first? | deviation proved? |
|---|---|---|---|
| **gemma-4-26B onA** | **yes** — `advisor_norm88` = advised `l1/width_sharded/1x88`, 88 cores | **yes**, candidate #1 | n/a — it shipped it, −12.98 % |
| **gemma-4-26B FN** | **yes** — `advisor_norm88` | **yes**, candidate #1 | **yes** — measured regression: 1.3469 vs 1.3412 sliding, 1.5499 vs 1.5394 full |
| **north-mini FN** | **yes** — `advisor_moe_norm_22` = advised `l1/width_sharded/1x22` | **yes**, candidate #1 | **yes** — 22 → 0.5432, its own 32 → 0.5184, 64 → 0.5733 |
| **north-mini B** | **yes** — `advisor_dense_chain_exact` | **yes** | **yes** — regression, 0.2341 vs 0.2033 incumbent |
| **qwen3-27B B** | **yes** — 4 advisor candidates (`advisor_rope_q_l1`, `_k_l1`, `advisor_qkv_direct`, `advisor_rope_dram`) | **yes** | **yes** — none beat the incumbent |
| **qwen3-27B FN** | **yes** — `rope_query_c32` / `rope_key_c32` / `rope_both_c32` = advised `32x1`, 32 cores | **no** — tried at #7–13, *after* the candidate it shipped at #3 | **yes** — measured regression, 1.2225 vs 1.2174 |
| **phi-3.5 onA** | **yes** — `rope_l1_rect32`, a 32-core height shard | **yes**, candidate #1 | n/a — it shipped it, −7.58 % |
| llama-3.1-8B exp17 | **partly** — `dense_geometry_64` = advised `1x64`; the advised norm (11 / 22) was never tried | — | **no** — the norm chain was filed `below_threshold` |
| llama-3.2-1B exp17 | **partly** — `dense_2_residual_chain_64`, `concat_output_dram` | — | partly |
| **phi-3.5 B** | **partly** — sharded `multiply`/`add` in the query's own layout; `neg`/`concat` left interleaved | yes | **no** — no reason recorded |
| **phi-3.5 FN** | **no** — shipped L1 interleaved throughout; the advised sharding was never tried | — | **no** — and §4 measures what it cost |
| phi-3.5 exp17 | **no** — `rope_l1_tail`, `rope_full_l1`, both interleaved-only | — | **no** |
| **gemma-4-12B exp11** | **no** — **52 measurements, not one of them an advised grid** (all L1-residency variants) | — | **no** |
| gemma-4-26B B | **no** — its own artefact says the *"advisor norm/grid direction was not booked into the shipped attention-boundary win"* | — | **no** |
| north-mini onA | **no** — nothing was tried; 3 incumbent measurements only | — | arithmetic recorded (`not_measurable`), but see the ceiling problem in §6 |

**Score: 7 of 15 cells tried the advisor's exact advice; 6 of those tried it first. Every one of those 7 has a
clean outcome — it either shipped the advice or measured a regression against it.** The protocol works when it
is followed.

**5 cells deviated with no recorded reason** (phi FN, phi B, phi exp17, gemma-4-12B, gemma-4-26B B). All five
deviated the *same way*: they took the advisor's L1 placement and dropped its sharding.

---

## 3. The deviation is always the same one

Of the 9 cells that changed anything, what they shipped versus what was advised:

| cell | advised (corrected) | shipped | strategy | cores |
|---|---|---|---|---|
| gemma-4-26B onA | `l1/width_sharded` 88 | 88-core width-sharded RMSNorm | ✅ | ✅ |
| phi-3.5 onA | `l1/height_sharded` 32 | 32-core rectangular height shard | ✅ | ✅ |
| phi-3.5 B | `l1/height_sharded` 32 | 32-core height shard on the 96-wide ops only | ✅ partial | ✅ partial |
| north-mini FN | `l1/width_sharded` 22 | 32-core width-sharded | ✅ | ❌ *proved better* |
| phi-3.5 FN | `l1/height_sharded` 32 | L1 **interleaved** | ❌ | ❌ |
| gemma-4-12B exp11 | grids on `rotary_embedding` (32, 16) and `rms_norm` (11, 64, 8) | L1 residency, no grid | ❌ | ❌ |
| qwen3-27B FN | `l1/height_sharded` 32 | one L1-interleaved conversion | ❌ | ❌ |
| gemma-4-26B FN | `l1/width_sharded` 88 | a boundary decision | ❌ | ❌ |
| gemma-4-26B B | `l1/width_sharded` 88 | DS matmul roles | ❌ | ❌ |

**The advisor's `isL1` call is followed widely. Its sharding is followed in 3 of 9, its exact grid in 3 of 9**
(and once knowingly beaten). That is `LayoutScore` level 1 being used and level 6 being discarded — except
that in the one cell where I could measure the difference directly, discarding it was expensive.

---

## 4. E25 — implementing phi FN's advice verbatim: **−10.43 %, against the −4.88 % it shipped**

phi FN is the one cell with the artefacts to settle this: a full `final_ir.mlir`, a decision trace, and a
before/after profile pair. I implemented the advised chain exactly as the IR specifies it — `%14`–`%22`,
including the two-range 32-core `CoreRangeSet` and the tile-aligned `(32,64)` / `(32,96)` shard shapes — and
measured four forms with the **cell's own unmodified `harness.py`**, one fresh process each, WARMUP 10 /
REPEATS 5 / ITERS 50.

| form | median ms | Δ vs incumbent | differential PCC | repeats |
|---|---|---|---|---|
| `off` — frozen incumbent | 0.807535 | — | — | 0.808512 … 0.807535 |
| **`interleaved` — what phi FN shipped** | 0.768104 | **−4.88 %** | 1.0 | 0.767374 … 0.768156 |
| `sharded_mul` — what phi B shipped | 0.751277 | **−6.97 %** | 1.0 | 0.751010 … 0.752433 |
| **`full` — the advisor's IR, verbatim** | **0.723320** | **−10.43 %** | **1.0** | 0.722631 … 0.723988 |

- The control reproduces the cell's own numbers: 0.807535 vs its recorded 0.807152; 0.768104 vs 0.767542.
- **Strict non-overlap between all three candidates:** max(`full`) 0.723988 < min(`sharded_mul`) 0.751010 <
  min(`interleaved`) 0.767374. The ordering is not noise.
- **Every form is bit-identical to the incumbent, PCC 1.0**, at the cell's own 0.999999 bar. There is no
  correctness objection to any of them.

**The advisor's exact advice is 2.1× the gain the stage shipped.** The cost of the unproven deviation is
**5.55 pp = 0.0448 ms/layer × 32 layers ≈ 1.43 ms/model**, on a cell that reported `improved` and passed its
gate.

phi B shipped the middle form; the same reasoning says it is leaving the `neg`/`concat` half on the table too,
though I have not re-measured on phi B's own incumbent (0.7886), so I am not attaching a number to it.

**Artefacts:** `~/skillexp-logs/exp-rope-faithful/` — `rope_off|rope_interleaved|rope_sharded_mul|rope_advisor_full.{json,log}`,
`oracle_modes.json`. Patch applied to a detached worktree of the cell branch; the cell's tree was not touched.

---

## 5. Retraction: the advisor did not recommend an illegal layout. I did.

I previously reported (old §3.23 / D6 / E24) that the advised `l1/height_sharded/32x1` could not run, on the
strength of two `TT_FATAL`s, and escalated it to *"the op model accepts a shard the runtime rejects — a
tt-mlir ↔ tt-metal validation gap"*. **All of that is withdrawn.**

**What the advisor actually specified**, from `final_ir.mlir`:

| | shard shape | cores |
|---|---|---|
| 48-wide (`neg`, the halves) | `memref<1x2x tile<32x32>>` = **(32, 64)** — tile-aligned, padded | 32 (two ranges) |
| 96-wide (`concat` out, `multiply`, `add`) | `memref<1x3x tile<32x32>>` = **(32, 96)** | 32 (two ranges) |

**What I implemented in the probe:** shard `(32, 48)` — the logical width, not the padded tile width — which
the advisor never specified anywhere. And I sharded the `concat` *output* while leaving its inputs interleaved,
where the IR shards the inputs first (`%18 = to_memory_config(%15) -> layout24`).

**Isolated single-op test of the advisor's exact config** (one op at a time, the real `CoreRangeSet`, on device):

| op | config | result |
|---|---|---|
| `ttnn.neg` (1,32,32,48) | shard (32,64), 32 cores | **OK** |
| `ttnn.concat` 2×(…,48) → (…,96) | shard (32,96), 32 cores | **OK** — the case I claimed was impossible |
| `ttnn.multiply` (…,96) | shard (32,96), 32 cores | **OK** |
| `ttnn.add` (…,96) | shard (32,96), 32 cores | **OK** |
| `ttnn.slice` → (…,48) | shard (32,64), 32 cores | **OK** |
| **control: `ttnn.neg` shard (32,48)** — my old probe | | **FAIL** — `TT_FATAL tensor_layout.cpp:162 !shard_align_error.has_value()` |

**The only thing that fails is the shape I invented.** Every op in the advisor's actual plan runs, and §4 shows
the whole plan is also the fastest and bit-identical. There is no validation gap, no illegal advice, and no
tt-metal bug here.

**Lesson, recorded because it changed a headline conclusion:** a `TT_FATAL` from *my* reconstruction of advice
is evidence about my reconstruction, not about the advisor. The order should be: read the IR, test the single
op in the exact advised config, and only then look at the optimizer.

---

## 6. What is still a real problem

Nothing above weakens these, and §1 and §4 sharpen them:

1. **The screening ceiling prices boundary conversions only**, so an in-chain re-grid is worth `0.000 µs` to it.
   Two of the corpus's three biggest wins came from cells whose ceiling said `0` / `not_measurable`
   ([`ADVICE-FOLLOWED`](ADVCHAL-V2-ADVICE-FOLLOWED.md)). That is the mechanism behind most of the
   never-tried rows in §2 — gemma-4-12B ran 52 measurements without trying an advised grid because no advised
   grid was ever priced above threshold.
2. **`advised_cores` is wrong on 58.3 % of advised ops** (§1) — a stage-side parse of a lossy field.
3. **`agrees_with_shipped` never compares the memory space**, and 23.2 % of op↔advice pairings are positional
   guesses that nothing downstream discounts.
4. **The differential oracle at 0.999999** still cost phi FN a −13.4 % combined candidate at PCC 0.9999911.
   Note §4's relevance: all three rope forms score exactly 1.0, so the oracle is not what blocked the sharding
   — nothing blocked it; it was never tried.

---

Generated from the cells' own artefacts. Measurements in `~/skillexp-logs/exp-rope-faithful/`,
chronology in `advchal-v2-chronology.json`, corrected advice in `advchal-v2-corrected-advice.json`.

---

## 7. E26 — apply the advised plan *together* instead of building chains up: −17.84 %

§4 implemented one advised block (the RoPE body) and got −10.43 %. The obvious next question: what if the cell
had applied **everything the advisor said**, rather than assembling chains one at a time?

phi FN's advised plan is 35 ops. Two blocks carry almost all of the cost the advisor disagreed about: the RoPE
body, and the two 44 µs `rms_norm`s it wants on `l1/block_sharded/1x11`. Both are implementable — the cell
already has a `advisor_norm_cores` knob. Measured with the cell's own harness, fresh process per configuration:

| configuration | median ms | Δ vs incumbent | differential PCC |
|---|---|---|---|
| incumbent — advisor ignored | 0.807535 | — | — |
| **what the cell shipped** — rope L1 interleaved | 0.768104 | **−4.88 %** | **1.0** |
| norm 11 only (the advised grid) | 0.747428 | −7.44 % | 0.99999107 |
| **rope advised only** | 0.723320 | **−10.43 %** | **1.0** |
| rope interleaved + norm 11 — the cell's own best, discarded | 0.700120 | −13.30 % | 0.99999107 |
| **rope advised + norm 11 — both advised blocks together** | **0.663507** | **−17.84 %** | 0.99999107 |

Strict non-overlap at every step: max(`ropefull+norm11`) 0.664564 < min(`ropeintl+norm11`) 0.699557, and
max(`rope advised`) 0.723988 < min(`shipped`) 0.767374.

**Applying the advisor's placement on both blocks is 3.7× what the cell shipped.**

### The oracle result matters here, and it is clean

| configuration | differential PCC vs frozen incumbent |
|---|---|
| rope interleaved, norm off | **1.0** |
| rope **advised**, norm off | **1.0** |
| norm 11, rope off | 0.9999910666979231 |
| norm 11 + rope interleaved | 0.9999910666979231 |
| norm 11 + rope advised | 0.9999910666979231 |

Two things fall out. **The advised RoPE sharding is bit-identical** — the entire −10.43 % is free, with no
correctness question of any kind. And **the PCC movement is exactly the same number whatever else changes**, so
it is attributable wholly to the norm re-grid and behaves like a deterministic reduction-order difference, not
like a shard-spec bug (a bug would interact with the other change).

**So even taking the skill's strict rule at face value — reject anything that moves PCC at all — phi FN should
have shipped −10.43 %, not −4.88 %.** The oracle is not what cost it the 5.55 pp; not trying the advice is.

---

## 8. Should "apply everything, then subtract" be the starting point?

**What the skill instructs today** (`SKILL.md` §4, verbatim):

> ### 4. Screen, in the order the reconciliation gives
> Each chain as one unit, one variable per measurement, against the frozen incumbent.

That is a **build-up** procedure: start from the incumbent, add one chain at a time. Nowhere does the skill ask
for the advised plan as a whole to be measured, and `final_ir.mlir` — the only artefact carrying the complete
plan with its shard shapes — is never mentioned.

**What the corpus shows.** Ranking the 9 cells that changed anything by outcome, against whether their *first*
candidate was the advisor's placement:

| cell | first candidate | Δ shipped |
|---|---|---|
| **gemma-4-26B onA** | **the advised plan** — `advisor_norm88`, its one dominant advised item | **−12.98 %** |
| **north-mini FN** | **the advised plan** — `advisor_moe_norm_22` | **−9.26 %** |
| **phi-3.5 onA** | **the advised geometry** — `rope_l1_rect32` | **−7.58 %** |
| phi-3.5 B | partly — the rope chain, sharded `multiply`/`add` | −5.09 % |
| phi-3.5 FN | no — rope L1 interleaved | −4.88 % *(−17.84 % was available)* |
| qwen3-27B FN | no — its own `packed_qkv_l1_chain`; the advised 32 was tried 4 candidates later | −2.25 % |
| gemma-4-12B exp11 | no — 52 measurements, not one an advised grid | −1.83 % |
| gemma-4-26B FN | the advised plan, and it **correctly rejected it** (its incumbent already had 8 cores, so this was 8→88, not 1→88) | −1.69 % |
| gemma-4-26B B | no — DS matmul roles; its own notes say the advisor's grid direction "was not booked" | −0.47 % |

**The top three outcomes in the corpus are the three cells whose first candidate was the advisor's placement.
The bottom four are cells that never applied it.**

⚠ **This is a correlation and I am not claiming it is wholly causal** — the cells also differ in how much
placement defect there was to find (gemma-4-26B onA had a 1-core norm; gemma-4-12B did not). What raises it
above correlation is that **for one cell the counterfactual is measured**: same cell, same frozen incumbent,
same harness — apply-all gives 3.7× what building up gave.

### Why build-up loses, mechanically

1. **A chain below the noise floor is never tested at all.** 60 % of the disagreed-on cost corpus-wide sits in
   `below_threshold` chains ([`ADVICE-FOLLOWED`](ADVCHAL-V2-ADVICE-FOLLOWED.md)). Individually unmeasurable;
   together, plainly measurable. phi FN's own norm chains were 178 µs and 196 µs and it took the whole-window
   combination to see them.
2. **The skill already knows this and only half-applies it.** Its `aggregate_only` feasibility verdict says
   *"apply the top chains together as one candidate first"* — but only as a fallback when no single chain clears
   the floor. The corpus says it is the better *default*.
3. **Build-up hides interactions.** rope + norm together (−17.84 %) beats the sum of the parts measured
   separately (−10.43 % and −7.44 % → −17.87 % predicted; here they happen to be additive, but nothing in the
   build-up procedure checks that, and the cell never got to a configuration where it could).
4. **"Not tried" and "tried and lost" become indistinguishable.** Four cells' unproven deviations all look
   identical in the artefacts to a measured rejection.

### The instruction change

**Make the advisor's complete plan, implemented from `final_ir.mlir`, candidate #1 — then ablate downwards.**

```
1. apply_all      = every advised placement at once, from final_ir.mlir (shard shapes + full CoreRangeSet)
2. if it does not run, remove ONLY the item that fails, with a single-op test naming it, and record that
3. measure. if apply_all wins, that is the floor, not the ceiling
4. ablate: remove one advised item at a time. an item whose removal makes it FASTER is a real finding
   about the advisor and should be reported as one
5. build up from the incumbent only for what apply_all could not reach
```

This inverts the failure mode. Today a cell that does nothing lands at "no_change" and passes its gate; under
apply-all-first, a cell that does nothing has to *explain a measured regression*. And step 4 is the part that
produces advisor feedback the current procedure structurally cannot: today an item that is never applied
generates no evidence about the advisor at all.

**Cost:** roughly the same device time. phi FN spent 12 measurements building up; apply-all + the ablations
that matter is the same order. It is a reordering, not more work.

→ [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) §F5.

---

## 9. What "apply everything" would have meant, per cell

Not measured except for phi FN — this is the advised plan each cell would have started from, from its own
corrected `report.json`, with what it actually did instead. **Analysis, not measurement.**

| cell | the advised plan's main items (corrected cores) | what it applied |
|---|---|---|
| **phi-3.5 FN** | `rms_norm`×2 → block_sharded **11**; rope body → height_sharded **32**; qkv `linear` → width_sharded **96**; `create_qkv_heads` → height_sharded **32**; SDPA + `concat_heads` → **DRAM**; MLP `linear` → **103**, `multiply` → **86**, down → **96** | rope, as L1 interleaved. **Measured: applying just the first two items = −17.84 %** |
| **gemma-4-12B exp11** | `rms_norm` → **11**, **64**, **8** by site; `rotary_embedding` → height_sharded **32** and **16**; `linear` → **96**, **60**, **86**; `multiply` → **96** | 52 measurements of L1-residency variants; **no advised grid ever applied** |
| **gemma-4-26B B** | `rms_norm` → width_sharded **88** and **8**; `linear` → **107**, **88**, **66**; `rotary_embedding` → height_sharded **16** | DS matmul roles. Its own notes: the advisor's norm/grid direction "was not booked" |
| **gemma-4-26B FN** | same plan as B | applied `advisor_norm88` first, measured a regression — **correct** |
| **gemma-4-26B onA** | same plan as B | applied `advisor_norm88` first → **shipped it, −12.98 %** |
| **north-mini FN** | MoE `rms_norm` → width_sharded **22**; `linear` → **80**, **96**, **64** | applied 22 first, then its own 32, shipped 32 with the comparison recorded — **correct** |
| **qwen3-27B FN** | rope `neg`/`multiply` → height_sharded **32**; `multiply` → **90**, **96**, **109**; `concat`/`rotary` → **DRAM** | its own packed-qkv candidate first; the advised 32 tried 4 candidates later and regressed |
| **llama-3.1-8B exp17** | `rms_norm` → block_sharded **11** *and* width_sharded **22**; `linear`/`add` → **64**, **90**, **96** | the advised **64** chain; the advised norm grids never tried (`below_threshold`) |
| llama-3.2-1B exp17 | `rms_norm` → **10**; `add` → **80**, **90** | a 64-core residual chain and a DRAM concat output |
| north-mini B / onA, phi exp17 | — | zeros; north-mini onA tried nothing at all |

**The pattern in one line: every cell whose advised plan was dominated by a starved reduction and which applied
it first, won big. Every cell that assembled its own chains instead is in the bottom half — and the one case
where the counterfactual is measured, it lost 3.7×.**
