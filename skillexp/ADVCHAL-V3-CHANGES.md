# Stage 02b `$advisor-challenger` — v3: what changed, how to test it, and what to expect

The companion to [`02b-v2-rebuild.md`](https://github.com/tenstorrent/agentic-research/blob/main/shard-advisor-experiments/03-advisor-stage-v2/analysis/02b-v2-rebuild.md).
That document recorded what v2 implemented; this one records what **v3** implements in response to the
15-cell v2 corpus, and what difference each change is expected to make.

It deliberately contains **no v3 measurements**. Everything quantified below is a v2 measurement, and the
point of the file is to be able to check, afterwards, whether the changes did what they were predicted to do.

| | |
|---|---|
| stage branch | tt-metal [`mvasiljevic/qb2/skillexp/challenger-skill-v3`](https://github.com/tenstorrent/tt-metal/tree/mvasiljevic/qb2/skillexp/challenger-skill-v3) |
| stage base | tt-metal `db00a4404ac` (= v2's shipped tree `3cb7d326930`), tagged `advchal-v3/base` |
| tracer branch | tt-mlir [`mvasiljevic/advchal-v3-ttnn-jit`](https://github.com/tenstorrent/tt-mlir/tree/mvasiljevic/advchal-v3-ttnn-jit) @ `97724a1170` |
| advisor pin | tt-mlir **`618cd4e75d` — unchanged**, tagged `advchal-v3/advisor-pin` |
| publish namespace | `advchal-v3` — v2's 14 `skillexp/done/advchal-v2/**` tags are untouched |
| files | `.agents/skills/advisor-challenger/{SKILL.md, scripts/{reconcile,harness_template,capture_template}.py}` · `.agents/prompts/model_bringup_multigoal/02b-advisor-challenger{.txt,.check.sh}` |

**Where v3 starts.** The whole change is one diff:

```
git diff advchal-v3/base..mvasiljevic/qb2/skillexp/challenger-skill-v3 -- .agents/
git diff advchal-v3/advisor-pin..mvasiljevic/advchal-v3-ttnn-jit          # tt-mlir, ttnn-jit only
```

Every v3 commit message begins `advchal-v3:`. Nothing before the base tag was touched.

**The optimizer did not move, on purpose.** The tt-mlir branch changes three files, all under
`tools/ttnn-jit/`, and **zero** files under `lib/` or `include/`. So a v2→v3 difference in outcome is method,
not tool — the property that made the v1/v2/v3 comparison readable in the first place. The three optimizer
defects the corpus found (`OPT-1` no latency term, `OPT-2` the normalization `coreCount` override, `OPT-3`
dedup before legality) are **not** addressed here; see [§4](#4-what-v3-deliberately-does-not-change).

---

## 1. The finding that drove the rebuild

v2's headline was that the harness decided every v1 outcome. **v3's is that the stage threw away most of what
the advisor gave it.** Of 17 defects the corpus catalogued, **ten are the stage mishandling a correct answer**,
each a one-file change with no build — and those cheap columns are where the value was:

- **It never applied the advisor's plan as written.** On the one cell where the counterfactual could be
  measured afterwards, the plan implemented verbatim was **−10.43 % at bit-identical output**, against the
  **−4.88 %** the cell shipped after 14 measurements; with the advised norm too, **−17.84 %**. **3.7×.**
- **Its screening metric prices its most reliable recommendation at `0.000 µs`.** The ceiling counts only
  boundary conversions, so re-gridding an op inside its L1 chain is worth zero to it — and **two of the three
  biggest wins in the corpus came from cells whose ceiling said zero and whose every chain read
  `below_threshold`**.
- **Its correctness rule rejected a candidate that was more accurate than what shipped.** "Reject if PCC moves
  at all" cannot tell a kernel bug from floating-point reassociation, which is *guaranteed* whenever a
  reduction's core count changes.
- **It read a lossy summary instead of the advice.** `report.json`'s `cores=` field prints the first range of a
  multi-range `CoreRangeSet`, so **58.3 % of advised core counts were understated and 34.4 % of the reported
  disagreement was phantom**; and it carries no shard shape at all, so a plan built from it cannot be
  implemented.

## 2. What was implemented

Each row names the v2 defect id it closes ([`READ-THIS`](https://github.com/tenstorrent/agentic-research/blob/main/shard-advisor-experiments/03-advisor-stage-v2/ADVCHAL-V2-READ-THIS.md) §3),
the action point it comes from ([`IMPROVEMENTS`](https://github.com/tenstorrent/agentic-research/blob/main/shard-advisor-experiments/03-advisor-stage-v2/ADVCHAL-V2-IMPROVEMENTS.md)),
and the file.

### 2.1 Apply the plan whole, then ablate — `STG-1` / `F5`

`SKILL.md`'s screening section is replaced. Candidate #1 is now **every advised placement at once**, built
from `final_ir.mlir`; `unfixable_ops` are dropped first; if what remains will not run, only the failing item
is removed, with an isolated single-op test naming it; then each advised item is **ablated** one at a time.
Build-up from the incumbent is what remains for whatever apply-all could not reach.

`final.json` must carry **`advised_plan_verbatim`** with a `measured_ms` or a `hard_error`, and the gate
fails without it. In v2, "not tried" and "tried and lost" were indistinguishable in the artefacts, and four
cells were in the first category with no reason recorded.

Ablation is the half that has no substitute: **an advised item whose removal is faster is feedback about the
advisor that a build-up order cannot generate at all.**

### 2.2 The correctness rule: absolute vetoes, differential observes — `STG-5` / `A1`

The absolute comparison at **the model's own bar** is the veto: candidate within the bar, **and no worse than
the incumbent against the same reference**. The differential comparison is recorded as an observation and
cannot veto. New required fields: `oracle_kind`, `oracle_pcc_bar`, `oracle_bar_source` (file:line),
`incumbent_pcc_vs_reference`. The gate refuses a bar tighter than any model's own test bar without a recorded
justification — the 0.999999 one cell invented is exactly what cost the corpus its largest measured win.

This is a **correctness** change, not only a permissive one: a differential oracle cannot tell which side
moved, and built absolutely for one discarded candidate the candidate scored **0.99931** where the **shipped
incumbent scored 0.98347**, failing the model's own 0.995 bar. The old rule can ship the less accurate
configuration and reject the better one. It did, twice.

### 2.3 Find the win before spending device time — `STG-2` / `C1b`, `B4`

`reconcile.py` emits **`cliff_candidates`**: material ops (≥2 % of the window) on ≤2 cores where the advisor
wants strictly more, ranked by per-model µs, each with its legal ladder. `screening_order` puts them second,
after the whole plan and before the chains. A new feasibility verdict **`regrid_only`** fires when the
boundary ceiling is under the floor but a cliff op exists, so a zero ceiling stops being a stop condition —
and the gate makes an **unscreened cliff candidate CRITICAL**, which is what closes the false-zero hole.

Over the corpus's own per-op data this rule flags 5 of 14 cells, contains all three double-digit wins, no
unflagged cell produced one, and it was used to *predict* an unscreened −12.44 %/layer win before measuring
it. Ranking by it moves the winning candidate to **rank 1 in all four** cells whose win was of this class,
from 2nd / 2nd / 2nd / **4th-of-27**.

### 2.4 Read the advice, not the summary — `STG-3`, `TOOL-2` / `C5f`, `F4`

`--ir final_ir.mlir` is now read and required by the gate. `reconcile.py` parses each layout's grid, **shard
shape** (from the memref tile dims), memory space and true `CoreRangeSet` size, emits them as
**`advised_plan`**, and takes `advised_cores` from the **grid-string product** rather than the `cores=`
bounding box. The grid product was validated 22/22, 10/10 and 17/17 against three decision traces where the
bounding box was right 2/22, 1/10 and 2/17.

Only the consumer is fixed. Changing `report.json` as well would be a double correction for no gain.

### 2.5 Honour what the advisor says it cannot do — `STG-4` / `C5g`

Ops in `unfixable_ops` (and those annotated `ttnn.validation_unfixable` in the IR) get their own accounting
bucket, are excluded from screening and from the disagreement accounting, and carry the advisor's exact
runtime error verbatim. The gate warns if one was measured anyway. **54 declarations in the corpus, 41
presented to a cell as screenable advice**, and cells spent device time rediscovering the identical string.

### 2.6 The legal grid ladder — `TOOL-3` / `B1`, `D4`

`reconcile.py --legal-grids <width-in-tiles>` prints the ladder, and every cliff candidate carries its own.
The rule is derived, not fitted: the shard-padding constraint `(C−1)·⌈W/C⌉ < W`, intersected with the exact
tile divisors and whole compute-grid rows (`--grid-row-width`, 11 on Blackhole, 8 on Wormhole). At W=64 it
reproduces the ladder measured on north-mini exactly — `{1,2,4,8,11,16,22,32,64}` — and excludes the
40/44/48/55/88 that hard-failed there and that one cell drew a conclusion from anyway.

Computed in the stage rather than in the advisor: same answer, no build, and it works on the Wormhole grid
without a second capture.

### 2.7 The impossible contract, closed — `D2` (v2 §) / `C1`

`reconcile.py --evidence` merges a separately-authored measurement record into a **freshly generated**
reconciliation, by stable identifier, with an unknown identifier fatal. v2 wrote `verdict: pending`, the gate
demanded it filled, and the skill forbade hand-editing — so four cells invented four ways out and were
effectively graded on which violation they chose. (The implementation is the `--evidence` variant one of those
four cells wrote, promoted to the supported path.)

### 2.8 Measurement protocol — `D8` / `C3`, `B3`, `C2`, `C2z`

- **`process_ordinal`** is recorded by the harness, counted in a session marker file, and the skill requires a
  throwaway `--label warmup_discard` process first. The first harness process of a session measured a floor
  of **11.838 µs** where the same configuration later measured **0.196 µs** — 60×, cross-process JIT-cache
  warmth that no per-process warm-up can remove, and the floor decides `feasibility.verdict`.
- **"Tighten the harness" is removed as advice for an overlap.** Measured: 250→1,800 replays made the floor
  **3–4× worse** and still did not separate the candidate. Drift does not average down.
- **Control-plus-one-knob** must be shown as a one-field policy diff, built with `dataclasses.replace`. The
  one cell that checked found it false and remeasured eight configurations.
- **Traced replay is stated as load-bearing**, with the reason: for the highest-yield candidate class host and
  device costs move in *opposite* directions (+45.6 µs eager, −65.2 µs traced), so **an eager harness would
  have rejected every norm win in the corpus.**

### 2.9 Provenance and legibility — `STG-6`, `STG-7`, `STG-9`, `STG-10`, `D13`

| change | why |
|---|---|
| a space mismatch between the advice and the op's input-0 memory is recorded as `space_hint`, and `agreed_on: grid \| ds_family` says what agreement rests on | **narrowed after step 0 measured it** — the profile has no output memory column, so input-0 space cannot decide agreement: as a bucket rule it moved 15 rows and only 1 was real ([`STEP0`](ADVCHAL-V3-STEP0.md)). The hint points at the IR edge instead |
| positional pairings excluded from anything presented as a **finding** (kept for cost) | 23.2 % of pairings corpus-wide are positional guesses the tool documented and then ignored; seven published claims were downgraded to undecidable |
| `capture_scope` — ops attempted, **methods substituted**, env knobs, `stopped_at` | 15 captures of 54–290 lines, uncompared; where a method is substituted the advice for that region is advice for the stand-in |
| `reachable_by_advisor` in `final.json` | 4 of the corpus's 7 zeros were coverage zeros and nothing said so |
| before/after `tt-perf-report` pair, full invocation, category split reported as layout-induced vs graph-structural | 1 of 15 cells kept a pair, which is why op-level verification is impossible for the rest — and why the largest coverage win still has no number |
| `op_under_test {name, incumbent_grid, candidate_grid, legal_ladder}` | two arms use the same knob name with different defaults, so "the 88-core candidate" is 1→88 in one and 8→88 in the other |
| `candidate_shape_assumptions` | the corpus's largest wins are batch-pinned by construction and `final.json` never said so |
| within-kind **products required** when two winners are disjoint | both cells that built one gained; only 2 of 15 did |
| DS-matmul rule restated: screen the buffer type, **not** the grid | "screen DS last because it never won" is v1-derived and v2 contradicts it; but widening a DS matmul measured **+65 % slower**, 1 win in 7 |
| re-advise after a topology change (~18 s), record which capture each candidate was screened against | cheaper than one harness measurement; the stage screened everything against one start-of-run capture |
| rank the profile's own conversion ops independently of the advice | the largest single number in the corpus — 191 ms/model of `retilize` — sat at an advisor ceiling of 0.000 µs, *correctly*, and no advisor-derived worklist can reach it |

### 2.10 Coverage — `TOOL-1`, `TOOL-4`

The tracer handlers are inherited (`sparse_matmul`, mutable-state `ttnn.copy`, `paged_fused_update_cache`,
`ones_like`, `pow`, `TracedTensor.__getitem__`, `repeat_interleave`). New in v3: `--help` stops advertising
`--tracer interception` as a general fallback, because it cannot trace any decoder using
`ttnn.rotary_embedding_hf` — TTIR has only `rotary_embedding_llama`, which needs a `trans_mat` operand the HF
op has no equivalent for. That cannot be fixed by porting, so the claim is narrowed instead.

`SKILL.md` and `capture_template.py` now say: **attempt the trace before recording a blocker.** Of the ten ops
cells recorded as blockers, four already had handlers and one did not exist at all — and `uncapturable.ops`
("the tracer cannot trace this") is a different question from `unfixable_ops` ("the advisor will not place
this").

## 3. Bugs found while implementing this, and fixed

Not on anyone's list. Found by running the tool against the corpus's own artefacts rather than by reading it.

| # | bug | consequence | fix |
|---|---|---|---|
| **B1** | `reconcile.py --ir` was declared *"accepted and ignored"* | the one artefact carrying the advice was never read | §2.4 |
| **B2** | a boundary's cost was split in half **unconditionally**, even with a chain on only one side | the other half was attributed to nothing. On phi `fuse-noadvise` the ceiling was 71.637 µs while the candidates meant to explain it summed to **60.06 µs** — 11.5 µs of value visible in the ceiling and in no candidate, which is exactly the suppression this stage's own bias rule forbids. Now sums to 71.635 | split only when both sides are chains |
| **B3** | `--layers-in-window N` was never checked against the profile | a declared N the profile did not contain divided `per_layer_window_us` by N, understating **every per-layer and per-model number** by that factor, silently | abort unless the op sequence really repeats N× |
| **B4** | the gate's advisory counter was dead code in 3 of 4 sections — `[ $? -ne 0 ]` after `python3 … \|\| fail=1` always reads 0 | the summary line reported *"PASSED with 0 advisory warning(s)"* with warnings on screen | the checks exit 2 for advisory-only, and the shell maps the code |
| **B5** | `capture_template.py` defaulted `TT_METAL_ROOT` to one developer's absolute path | the shipped template only worked on one host | `TT_METAL_HOME`, then cwd |
| **B6** | the `DRAM Sharded` column was looked up once **per CSV row** | none, but it is a per-row scan of the header | hoisted |

**And one change I made, tested, and reverted** — recorded so it is not re-proposed. The corpus disproved
`nlp_create_qkv_heads_decode` on 1 core as a defect (it height-shards over *batch*, so its core count **is**
the batch size) and suggested a one-line check against `decode_batch`. Implemented as a filter, it deleted
**every cliff candidate on every batch-1 cell** — at batch 1 the test `shipped_cores == decode_batch` is
vacuous, and an op can only reach the starved list at ≤2 cores, so the filter can never fire where it would be
informative. It is a caution in `SKILL.md` prose instead.

## 4. What v3 deliberately does not change

The corpus's own base rate is that **about 1 in 6 of its recommendations was later refuted by measurement**, so
this section matters as much as §2.

| not done | why |
|---|---|
| **`OPT-1` a latency term in `LayoutScore`; `OPT-2` the norm `coreCount` override; `OPT-3` dedup before legality** | C++, needs a build — and it would change the advisor pin, so a v2→v3 comparison would conflate method with tool. The measured case is also weak: a ladder sweep captures the value the advised grid misses (82 % → ~99 %) without touching the objective. Revisit **after** v3, with the pin as the only variable |
| **the *"pick the legal grid closest to 16 cores"* heuristic** | it scored 99.4 % of achievable against the advisor's 82 % — on **three cells**. That is a fitted constant on n=3. v3 encodes the *mechanism* it approximates (the win is the first step off one core; sweep the ladder) and no constant |
| **a movement budget that fails the gate above ~10 %/25 % non-compute** | the right answer on 14 of 14 cells, and still two thresholds fitted to one corpus. Reported, never gated |
| **corpus median op-costs shipped in the skill for the cross-model outlier lens** | it found the corpus's largest un-screened defect (a 23× wrong-op call) and it needs cross-cell data a single cell does not have. Baking a table into a skill guarantees it goes stale — so this is a **post-run analysis step** over the v3 corpus, not a stage rule |
| **`report.json` emitting the full `CoreRangeSet`** | the consumer fix makes it cosmetic, and doing both risks a double correction |
| **`D4` the ladder emitted by the advisor** | computed in the stage instead: same answer, no build, works on any grid width |
| **mandatory N≥2 multi-layer capture** | `spill.ran` was true in 8 of 8 cells and 2–3 layers spill more. Narrowed to **one kind per cell**, on the kind that spills least, with `layers_in_window_reason` when none qualifies. It answers one question and does not cost every cell a re-capture |
| **`E-1` the 191 ms `retilize`; `E1a` sharded GQA SDPA output; the `concatenate_heads` wrong op** | none is a placement problem. v3 *reports* them; fixing them belongs to the decoder and to tt-metal |
| **relaxing non-overlap or the fresh-process confirmation** | v3 admits more candidates and loosens the oracle, so the false-positive controls must stay exactly as they are |

## 5. Risks

| # | risk | mitigation |
|---|---|---|
| **R1** | **apply-all hits L1 capacity.** `spill.ran` was true in 8 of 8 cells at one layer, so the whole plan may not fit | step 3 of the procedure: remove only the failing item, with the single-op test that isolates it, and record it. A capacity failure is a partial-application artefact, not a wrong direction |
| **R2** | **the absolute oracle needs a reference some cells do not have.** gemma-4-26B's real weights are absent from this host (28 KB, config only) | use the model's own bfloat16 `FunctionalDecoder` + `synthetic_layer_state_dict` — that is what settled gemma-4-26B `nofuse-noadvise` in the v2 analysis. Give each decoder a fresh KV cache |
| **R3** | **more candidates × a looser oracle manufactures false positives.** Best-of-N at a fixed per-comparison bar is not an N-independent risk | non-overlap at n=5 and the fresh-process confirmation are unchanged, and the cliff ranking *shrinks* the list (27 → 2 on one cell) rather than growing it |
| **R4** | **the legal ladder is advisory** — the op's own rulebook is not modelled | a rung outside expectation gets an isolated single-op test before a conclusion |
| **R5** | **the new cliff gate could block a legitimate zero** | a quoted `hard_error` discharges it, same as `material_ops_on_le_2_cores` |
| **R6** | **v3 changes the stage *and* the tracer**, so a v2→v3 delta on the four coverage-blocked cells mixes two causes | those cells' reconciliations are computable with and without the handlers; record both, and attribute the coverage half separately |
| **R7** | **the gate grew ~12 required fields.** v2's own history includes one extra required env var breaking every in-stage check | the CRITICAL/ADVISORY split holds: the five load-bearing fields are CRITICAL, the bookkeeping ones WARN. Verified below |

**Meta-risk, unchanged from v2 and worth restating:** everything derives from 15 cells, one architecture, one
host, decode only, mostly batch 32 or batch 1, and one advisor commit. Nothing is tested on Wormhole.

## 6. How to test it

### Step 0 — no device: run v3 `reconcile.py` over all 15 v2 cells' committed artefacts

This is the cheap check that the rewrite is right, and it is the same move v2 made. For each cell, with
`--ir`, assert:

1. **the accounting still closes to 100 %** of the same window as v2, and the window is byte-identical;
2. **`advised_cores` rises** on the ops whose `CoreRangeSet` has more than one range, and **`chain` rows
   become `agrees_with_shipped`** where the corrected count matches — the corpus predicts ≈**58 %** of advised
   ops understated and ≈**34.4 %** of chain µs turning out phantom;
3. **`advisor_unfixable` is non-empty in all 15** (`nlp_concat_heads_decode` appears in every cell);
4. **`cliff_candidates` flags 5 of 14 cells**, and contains phi `fuse-noadvise`'s two 1-core norms,
   gemma-4-26B `nofuse-noadvise`'s residual norm, gemma-4-26B `nofuse-noadvise-onA`'s four 1-core norms and
   north-mini `fuse-noadvise`'s MoE norm;
5. **the ceiling equals the sum of the chains' attributable value** (B2), where v2 left a gap;
6. `--self-test` is **21/21**.

**DONE — results in [`ADVCHAL-V3-STEP0.md`](ADVCHAL-V3-STEP0.md).** 21 of 26 kind-runs across 12 of 15 cells
replay (5 are not reproducible from what the corpus published: three cells committed no perf CSV, and
gemma-4-26B `nofuse-noadvise`'s CSVs do not reproduce its window — which means step 2's 26× prediction cannot
be pre-checked). All six predictions hold, at 56.0 % of advised core counts understated against 58.3 %
predicted, 18.4 % of chain rows carrying 31.1 % of chain µs turning out phantom against 17.7 % / 34.4 %, the
cliff check flagging 5 of 12 cells with three of the four win cells among them, the ceiling gap closing from
83.1 µs to 0.003 µs, and **seven feasibility verdicts moving off `not_measurable` on the two cells that
published zeros over screenable cliffs.** Step 0 also found and fixed one defect in v3 itself — see §2.9.

*Detail for two cells, from while implementing:* On phi `fuse-noadvise`:
window 725.175 µs unchanged and still closing; 14 of 35 advised ops understated; the wrongly-agreeing
`typecast` row moved out of `agrees_with_shipped`; `nlp_concat_heads_decode` moved into `advisor_unfixable`;
the ceiling now reconciles to 71.635 against 71.637; and the two 1-core `rms_norm` ops the cell discarded rank
**1 and 2** at 1424 and 1417 µs/model with the ladder `{1,2,3,4,6,8,11,12,16,24,32,48,96}` printed beside
them. On gemma-4-26B `nofuse-noadvise-onA` sliding — the cell that recorded `not_measurable` with a
`ceiling_us` of 0 and then shipped **−12.98 %** — v3 ranks its four 1-core norms first at ~1,100 µs/model each,
with the ladder `{1,2,4,8,11,22,44,88}`, which contains the **44** that later measured better than the advised
88.

**Gate verified end to end** on a synthetic well-formed cell: passes with **zero warnings**, and fails on
exactly the removed item — CRITICAL for `oracle_kind`, `oracle_pcc_bar`, `oracle_bar_source`,
`incumbent_pcc_vs_reference`, `advised_plan_verbatim`, a missing `--ir`, an unscreened cliff candidate and a
`0.999999` bar; ADVISORY for `reachable_by_advisor`, `candidate_shape_assumptions` and the perf-report pair.

### Step 1 — the stop-and-reassess cell: phi-3.5 `fuse-noadvise`, batch 32

The only cell whose outcome is **computed in advance**, because the whole ladder was measured on its own
harness afterwards. v3 must ship at least the advisor's plan:

| what | v2 | v3 must reach |
|---|---|---|
| shipped | −4.91 % / layer | **−17.84 %** (rope as advised + the advised 11-core norm) |
| the plan alone | never tried | **−10.43 %**, at differential PCC exactly **1.0** |
| model level | −1,267 µs | **−4,609 µs** |

**If v3 does not reproduce ≥ −10.43 % here, stop.** The failure is in the rebuild, not in the advisor, and
steps 2 onwards are not worth running. Two mechanisms have to fire and both are checkable in the artefacts:
`advised_plan_verbatim` measured (§2.1), and the norm shipping under an absolute oracle at 0.995 rather than
being rejected at 0.999999 (§2.2).

### Steps 2–8, in decreasing expected value

| # | cell | v2 | v3 expected | mechanism |
|---|---|---|---|---|
| 2 | **gemma-4-26B `nofuse-noadvise`**, b1 | −0.34 % sliding (−147.9 µs/model) | **−12.44 % sliding, ≈−3,918 µs/model — 26×** | cliff check + `regrid_only`; and the candidate is *more* accurate than what shipped (0.99931 vs 0.98347) so A1 admits it |
| 3 | **gemma-4-26B `-onA`**, b1 | −12.98 % sliding (norm 88) | **−13.63 %** — 44 cores, bit-identical, −375 µs/model | ladder swept below the advised value |
| 4 | **north-mini `fuse-noadvise`**, b1 | −10.23 % (32 cores) | **−11.28 %** — 16 cores, −264 µs/model | ladder `{1,2,4,8,11,16,22,32,64}` swept below the advised 22 |
| 5 | **llama-3.1-8B `exp17`**, b32 | 0.0 %, **the only verified real zero** | **0.0 %** | ⚠ **negative control.** Its whole ladder was swept and nothing beats the default. A "win" here means v3 loosened something it should not have |
| 6 | **north-mini `nofuse-noadvise`**, b1 | 0.0 %, ceiling *below* its floor | **> 0 screened candidates**; 11 worth 632 µs/model are visible | tracer handlers + `regrid_only`. Sign of the shipped result unknown — the prediction is that it stops being an unexamined zero |
| 7 | **north-mini `-onA`**, b1 | 0.0 %, sparse MoE untraceable | `untraced` **77.15 % → 14.39 %**, 2 candidates, 61.9 µs/model | tracer handler |
| 8 | **qwen3.6-27B `nofuse-noadvise`**, b32 | 0.0 %, 63.5 % of its dominant layer untraced | `linear_attention` captures in full (**69 advised ops**), and the conversion ranking must surface **3,983 µs/layer of `retilize` = 191 ms/model, 24 % of decode time** | the highest-value single measurement left in the corpus. Expect the *shipped* result to stay near zero and the *finding* to be the 191 ms — reported, handed off, not screened |
| 9 | phi `exp17`, gemma-4-12B `exp11`, llama-3.2-1B `exp17`, qwen `fuse-noadvise`, phi `-onA`/`nofuse-noadvise`, gemma-4-26B `fuse-noadvise`, north-mini's remaining kinds | as recorded | ≥ v2 everywhere; gemma-4-12B additionally must report the 1-core `concatenate_heads` outlier (**102.6 µs, 23× the corpus mean for the same logical step, ≈2.4–2.6 ms/model**) | apply-all-first; the conversion/outlier ranking. gemma-4-12B ran **52 measurements without trying one advised grid** |

**Cost.** v2 was 7 h 21 m, 316 M tokens, $53.00 for 149 device measurements. apply-all plus ablation plus the
ladder sweeps is roughly **1.5–2×** that: budget **12–15 h** and **$80–110**. Two of v3's biggest levers — the
cliff check and the ladder — cost **no device time at all**.

### What would falsify the rebuild

Stated in advance so the comparison is not retrofitted.

- phi `fuse-noadvise` does not reach −10.43 % → §2.1 or §2.2 does not work.
- llama-3.1-8B `exp17` ships a change → the loosened oracle is admitting noise.
- a cliff-flagged op is screened and regresses in **more than one** cell → the detection rule is not the
  threshold effect it is claimed to be. (One regression is expected: the same candidate regressed on
  gemma-4-26B `fuse-noadvise`, correctly, because that cell's incumbent already sat at 8 cores rather than 1.)
- the corrected `advised_cores` does **not** turn ≈⅓ of chain µs into agreement → the grid-product reading is
  wrong, and §2.4 has introduced an error rather than fixed one.

---

## 7. Where this file goes

`skillexp/` on the stage branch is where v2's results lived while its run was in flight, and this follows
that. When the v3 run completes it should be folded into agentic-research as
`shard-advisor-experiments/04-advisor-stage-v3/`, beside the three rounds it continues, with the v3
measurements added and the predictions in §6 marked hit or missed.

---

## 8. One environment item this run needs first

`/home/mvasiljevic/tt-metal/.git` cannot take new loose objects or new tag directories as `ttuser`: **175 of
its 225 `objects/` fan-out directories, and `refs/tags/skillexp/`, are owned by `root` at mode 755** — created
by a container that ran git as root. Any new object whose hash lands in one of those prefixes fails with
`error: insufficient permission for adding an object to repository database`, which is why v3's own commits
had to be built in a side clone and fetched in as a pack, and why the base tag is `advchal-v3/base` rather
than `skillexp/advchal-v3/base`.

**The drivers will hit this**, because they tag each finished cell `skillexp/done/advchal-v3/<arm>/<model>` —
a new directory inside the root-owned `refs/tags/skillexp/`. Fix before starting the run:

```
sudo chown -R ttuser:ttuser /home/mvasiljevic/tt-metal/.git
```

It is pre-existing and unrelated to v3; nothing below depends on it except the ability to write.
