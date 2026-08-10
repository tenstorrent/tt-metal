# advchal-v3 step 0 — the v3 tool replayed over the v2 corpus, no device

[`ADVCHAL-V3-CHANGES.md`](ADVCHAL-V3-CHANGES.md) §6 sets step 0 as the gate before any hardware time: run the
v3 `reconcile.py` over every v2 cell's own committed artefacts and check six predictions recorded in advance.
This is the result. **Five of six hold; the sixth held and then found a defect in v3 itself, which is fixed
here.** Verdict: proceed to Phase B.

Nothing here needed a device, and nothing outside a scratch directory was written.

## Method, and why the inputs were matched rather than named

For each `(cell, layer kind)` the corpus published a `reconciliation_<kind>.json`. The replay re-runs the v3
tool on the inputs that produced it and diffs the two.

**The perf CSV is identified by reproducing the window v2 recorded, not by filename.** Filenames vary across
cells (per kind, per scope, some under `invalid_prior/`), and pairing the wrong CSV would produce a confident
wrong comparison — the failure mode the v2 analysis hit repeatedly. If no committed CSV reproduces the window
to 0.011 µs, the pair is reported **not reproducible** rather than guessed at.

The incumbent is matched the same way, on its `incumbent_ms` rather than its name. That was not the first
attempt: naming `incumbent_<kind>.json` picked north-mini `fuse-noadvise`'s `incumbent.json`, whose
`incumbent_ms` is **27.63 ms** — a *derived full-model composite*, against a 0.55 ms layer — and the v3
window-ratio guard correctly refused all three of its kind-runs. That is the guard doing its job on an
artefact the v2 skill had warned about (`harness_scope`, "say whether the metric is measured or derived"), and
the cell in question is the corpus's weakest by three independent measures. Runner: `run_step0.py`.

## Coverage — 21 of 26 kind-runs, 12 of 15 cells

| not reproducible | why |
|---|---|
| gemma-4-26B `fuse-noadvise`, both kinds | **no perf CSV committed anywhere in the cell.** Already known — this is the cell missing from `advchal-v2-data.json` |
| gemma-4-26B `nofuse-noadvise`, both kinds | 4 CSVs present, none reproduces either window: closest is **1202.96 µs** against 1211.136 and 1206.737. The committed CSVs are not the ones the reconciliation was computed against |
| llama-3.2-1B `exp17` | no perf CSV committed |

**Consequence for the run plan.** gemma-4-26B `nofuse-noadvise` is step 2 — the **−12.44 % / 26×** prediction —
and step 0 cannot pre-verify its cliff candidate, because its own window is not reproducible from what it
published. That prediction stands unchecked until the device run. Worth knowing before relying on it; it is
also consistent with that cell being one of the two with no `done` tag.

## The six predictions, scored

### P1 — the window is unchanged and the accounting still closes ✅

**21/21** kind-runs reproduce v2's window to within 0.011 µs, and **21/21** still partition to 100 %. So
nothing below is a different measurement of a different thing.

### P2 — the corrected `advised_cores` turns about a third of the "disagreement" into agreement ✅

| | v3 measured | corpus predicted |
|---|---|---|
| advised ops whose `cores=` bounding box understates the true count | **191 of 341 = 56.0 %** | 58.3 % |
| v2 `chain` rows that agree once corrected | **51 of 277 = 18.4 %** | 59 of 334 = 17.7 % |
| …carrying | **1,360.0 of 4,368.5 µs = 31.1 %** of chain µs | 34.4 % |

Independently derived, on a smaller population (21 kind-runs rather than 26), and it lands within a couple of
points on all three. The phantom third of the disagreement is real.

### P3 — `unfixable_ops` is honoured, and it was being screened ✅

**17 of 21** kind-runs carry a declaration. **29 rows carrying 82.4 µs move out of `dram_resident`**, where
the reconciliation had labelled them *"advisor placed it in DRAM — that is advice"*. Two ops account for all
of it: `nlp_concat_heads_decode` (15 kind-runs) and `rotary_embedding` (7). The corpus's figure was 41 of 54
declarations presented as screenable across all 15 cells; 29 over the 12 reproducible ones is consistent.

### P4 — the cliff check flags 5 cells, and they are the right ones ✅

**5 of 12 cells**, exactly the 5-of-14 rate predicted:

| cell | cliff ops | what the corpus says about it |
|---|---|---|
| gemma-4-26B `nofuse-noadvise-onA` | **8** | shipped −12.98 % on exactly this class; 44 cores later beat the advised 88 |
| north-mini `nofuse-noadvise-onA` | 3 | published **0.0 %** |
| north-mini `fuse-noadvise` | 2 | shipped −10.23 %; 16 cores later beat the shipped 32 |
| gemma-4-12B `exp11` | 2 | 52 measurements, no advised grid among them |
| phi-3.5 `fuse-noadvise` | 2 | the two 1-core norms it discarded — rank **1 and 2**, 1424 and 1417 µs/model |

Three of the four win cells are here (the fourth is the unreproducible gemma-4-26B `nofuse-noadvise`), plus
the two cells that published zeros over screenable cliffs.

### P5 — the ceiling now reconciles to the candidates it is supposed to explain ✅

The B2 bug fix, at corpus scale: across these 21 kind-runs the ceiling exceeded the sum of its chains'
attributable value by **83.1 µs** under v2 and by **0.003 µs** under v3. **21/21** reconcile. That is 83 µs of
attributable value that v2 displayed in the ceiling and offered in no candidate.

### P6 — a zero boundary ceiling stops being a stop condition ✅ — the most consequential result

Seven verdicts change, every one in the direction of "screen it":

| cell / kind | v2 | v3 | boundary ceiling | cliff pool |
|---|---|---|---|---|
| north-mini `fuse-noadvise` full MoE | `not_measurable` | **`regrid_only`** | 0 | 26.1 µs |
| north-mini `fuse-noadvise` sliding MoE | `not_measurable` | **`regrid_only`** | 1.1 | 26.1 |
| north-mini `-onA` dense | `not_measurable` | **`regrid_only`** | 1.7 | 26.0 |
| north-mini `-onA` full sparse MoE | `not_measurable` | **`regrid_only`** | 0.6 | 26.1 |
| north-mini `-onA` sliding sparse MoE | `not_measurable` | **`regrid_only`** | 1.7 | 26.0 |
| gemma-4-26B `-onA` full | `not_measurable` | `aggregate_only` | 0 → 6.0 | 176.4 |
| gemma-4-26B `-onA` sliding | `not_measurable` | **`measurable`** | 0 → 2.4 | 176.9 |

**north-mini `-onA` published 0.0 % and gemma-4-26B `-onA` recorded `not_measurable` and then shipped
−12.98 %.** Under v3 neither is allowed to publish a zero without screening the cliff first — and the gate
makes an unscreened cliff candidate CRITICAL. This is the false-zero hole closing, measured on the cells that
fell into it.

## The defect step 0 found in v3, and the correction

P2's row-level diff showed **15 rows leaving `agrees_with_shipped` for `chain`** where the corpus documented
exactly one such row. That asymmetry is what made me look.

**Only phi `fuse-noadvise`'s `typecast` was real.** The other 14 were `linear` (6) and
`nlp_create_qkv_heads_decode` (6) plus a `rotary_embedding` and one more, all of the same shape: the advisor
places the op's **output** in L1, and `Input 0 Memory` says its **input** is in DRAM. Those are not in
conflict — a matmul legitimately reads DRAM and writes L1 width-sharded. And the profile has **no output
memory column at all**, which the tool's own `limitations[]` already said.

So C5c as I implemented it used the wrong quantity, and six of its 14 false positives were on the matmul
class a ladder sweep is already known to lose on (+65 % slower, 1 win in 7) — it would have manufactured
expensive candidates, 172 µs the largest.

**Fixed:** agreement is decided on grid or DS family as before, and a space mismatch is recorded as
`space_hint` — a pointer to check that edge in `final_ir.mlir` — not a rebucketing. The self-test fixture
that asserted the old behaviour now asserts the new one and carries the reason. `--self-test` is 21/21.

This is a genuine correction to [`IMPROVEMENTS`](https://github.com/tenstorrent/agentic-research/blob/main/shard-advisor-experiments/03-advisor-stage-v2/ADVCHAL-V2-IMPROVEMENTS.md)
C5c, whose fix — *"require the space to match as well"* — is not implementable from the profile. The finding it
rests on (1 of phi's 2 agreement rows is wrong) is correct; the remedy needs the IR.

It is also, precisely, the error pattern that file's own pitfalls document warns about most: reading a
summary field as if it were the authoritative artefact. Caught here at zero cost, which is the argument for
step 0.

## What this establishes, and what it does not

**Establishes.** The v3 tool reads the same windows, closes the same accounting, and changes the buckets in
the direction and roughly the magnitude predicted, on 12 independent cells. The three cheapest high-value
changes — the corrected core count, the honoured `unfixable_ops`, and the cliff check with its new verdict —
all behave as designed on real data. One defect in v3 was found and fixed.

**Does not establish.** Nothing here is a measurement: every number is a re-derivation over profiles that
already existed. Step 0 cannot tell whether screening a cliff candidate makes a decoder faster — that is
step 1, and phi `fuse-noadvise` must reach −10.43 % or the rebuild is wrong. It also cannot check
gemma-4-26B `nofuse-noadvise`'s 26× prediction, for want of a reproducible window, and it says nothing about
the four coverage-blocked cells, whose re-captures need the device.

Raw output: `run_step0.py` and `step0-results.json` (26 rows, one per kind-run, with the per-cell diffs).
