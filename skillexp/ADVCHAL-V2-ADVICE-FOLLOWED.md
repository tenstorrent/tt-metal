# How much of the advisor's advice was actually followed — all 15 cells

The per-op follows-advice work in [`PHI-BEFORE-ADVISED-AFTER`](ADVCHAL-V2-PHI-BEFORE-ADVISED-AFTER.md)
covered **one** cell. This does the corpus. The unit is the *chain* — the group of ops the stage actually
measures and ships (see §3.24 of [`READ-THIS`](ADVCHAL-V2-READ-THIS.md) for what `chain` means).

> ⚠ **Corrected after this file was first written.** The `advised_cores` field every row below was compared
> against is **wrong on 58.3 % of advised ops** — `reconcile.py` parses `cores=` out of `report.json`, which
> prints only the first range of a multi-range `CoreRangeSet`. The corrected counts (grid-string product,
> validated against three decision traces) move the geometry verdict from *"once in fifteen cells"* to
> **3 of the 9 cells that changed anything**, and two of those had recorded themselves as *overriding* the
> advisor. See [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §1, and the corrected table below.

## The headline

**58 of 248 chains were kept, carrying 589 µs of the 5,547 µs the stage counted as disagreed-on — 10.6 %. Corrected for the `advised_cores` defect, 589 of 3,639 µs = 16.2 %** (see the ladder below).

| chain verdict | chains | µs | share of disagreed µs | what it means |
|---|---|---|---|---|
| `below_threshold` | 108 | 3325.9 | 60.0 % | **dismissed without a measurement** |
| `rejected` | 55 | 1063.8 | 19.2 % | implemented, measured, lost or vetoed |
| `kept` | 58 | 589.3 | 10.6 % | the advice shipped |
| `not_measurable` | 26 | 568.4 | 10.2 % | the cell's ceiling was under its noise floor |
| `hard_error` | 1 | 0.0 | 0.0 % | implementing it hit a `TT_FATAL` |

Chains cover the `chain` bucket only. A further **86 ops / 1182 µs** sit in
`dram_resident` — *"leave this in DRAM"* is advice too, and it gets no chain at all. Against the full
disagreement (chain + dram_resident = 6729 µs) the followed share is
**8.8 %**.

### Corrected denominator: the disagreement was inflated by a third

The 5,547 µs above is the stage's own figure, and it is too big. `advised_cores` is understated on 58.3 % of
advised ops (§1 of [`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md)), which pushes rows into the
`chain` bucket that actually **agree** with the shipped placement. Re-running `reconcile.py`'s own test with the
corrected counts:

| | chain rows | chain µs |
|---|---|---|
| as the stage reported | 334 | 5,547.4 |
| **stop being disagreements once corrected** | **59** | **1,908.4 (34.4 %)** |
| genuine disagreements | 275 | **3,639.0** |

The phantom rows are `nlp_create_qkv_heads_decode` (advised 32, shipped 32 — in every cell),
`rotary_embedding` (16 or 32), several `linear`s (80/86/96/103/107), and every `slice_static` advised
`l1/interleaved` — those carry no `cores=` field at all, so `advised_cores` is `None` and the row can never
register as agreement even at a matching 110 cores.

**So the followed share is a ladder, not a number:**

| denominator | followed |
|---|---|
| chain µs as the stage reported it — 5,547.4 | 589.3 → **10.6 %** |
| **chain µs with corrected core counts — 3,639.0** | 589.3 → **16.2 %** |
| plus `dram_resident` (1,182 µs), *before* removing advisor-declared-unfixable ops | 589.3 → 12.2 % |

**16.2 % is the honest figure for "of the placement the advisor genuinely disagreed about, how much shipped."**
And the `dram_resident` bucket cannot simply be added to the denominator, because 41 of 54
advisor-declared-`unfixable` rows sit in it — for those the DRAM layout is a fallback after a declared failure,
not advice (§11).

⚠ **That is an upper bound.** `kept` means the *chain shipped*, not that the advised geometry was
implemented. phi FN shows the difference: of its 12 kept chains, 6 took only the buffer type (DRAM→L1) and
left the advised sharding unimplemented — because it does not run (§3.23).

## Per cell

| cell / layer kind | outcome | Δ | chains | kept | rej | `<thr` | n/m | kept µs | advised µs | followed |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-12B exp11 / `full_attention` | improved | -1.83 % | 14 | 8 | 0 | 6 | 0 | 305.0 | 397.8 | **77 %** |
| gemma-4-12B exp11 / `sliding_attention` | improved | -1.83 % | 14 | 6 | 2 | 6 | 0 | 50.4 | 333.5 | **15 %** |
| gemma-4-26B B / `full_attention` | SHIPPED | -0.47 % | 11 | 0 | 5 | 6 | 0 | 0.0 | 388.8 | 0 % |
| gemma-4-26B B / `sliding_attention` | SHIPPED | -0.47 % | 7 | 2 | 0 | 5 | 0 | 0.0 | 377.3 | 0 % |
| gemma-4-26B FN / `full_attention` | improved | -1.69 % | 11 | 1 | 0 | 10 | 0 | 0.0 | 295.7 | 0 % |
| gemma-4-26B FN / `sliding_attention` | improved | -1.69 % | 7 | 1 | 0 | 6 | 0 | 0.0 | 220.5 | 0 % |
| gemma-4-26B onA / `full_attention` | improved | -12.98 % | 4 | 0 | 0 | 4 | 0 | 0.0 | 458.3 | 0 % |
| gemma-4-26B onA / `sliding_attention` | improved | -12.98 % | 3 | 0 | 0 | 3 | 0 | 0.0 | 378.1 | 0 % |
| llama-3.1-8B exp17 / `dense` | contribution_zero | +0.00 % | 5 | 0 | 2 | 2 | 1 | 0.0 | 118.6 | 0 % |
| llama-3.2-1B exp17 / `dense` | no_change | +0.00 % | 4 | 0 | 2 | 2 | 0 | 0.0 | 65.9 | 0 % |
| north-mini B / `dense_full_forced_rope` | no_change | +0.00 % | 7 | 0 | 5 | 2 | 0 | 0.0 | 85.1 | 0 % |
| north-mini B / `full_no_rope_moe` | no_change | +0.00 % | 3 | 0 | 0 | 2 | 1 | 0.0 | 19.4 | 0 % |
| north-mini B / `sliding_rope_moe` | no_change | +0.00 % | 5 | 0 | 3 | 2 | 0 | 0.0 | 23.3 | 0 % |
| north-mini FN / `dense_full_attention` | improved | -9.26 % | 8 | 0 | 0 | 7 | 1 | 0.0 | 16.0 | 0 % |
| north-mini FN / `full_attention_moe` | improved | -9.26 % | 1 | 0 | 0 | 0 | 1 | 0.0 | 70.0 | 0 % |
| north-mini FN / `sliding_attention_moe` | improved | -9.26 % | 3 | 0 | 0 | 0 | 3 | 0.0 | 73.8 | 0 % |
| north-mini onA / `dense_full_attention` | no_change | +0.00 % | 6 | 0 | 0 | 0 | 6 | 0.0 | 235.0 | 0 % |
| north-mini onA / `full_attention_sparse_moe` | no_change | +0.00 % | 3 | 0 | 0 | 0 | 3 | 0.0 | 112.5 | 0 % |
| north-mini onA / `sliding_attention_sparse_moe` | no_change | +0.00 % | 5 | 0 | 0 | 0 | 5 | 0.0 | 116.8 | 0 % |
| phi-3.5 B / `dense` | improved | -5.09 % | 20 | 12 | 0 | 8 | 0 | 73.8 | 332.6 | **22 %** |
| phi-3.5 FN / `dense` | improved | -4.91 % | 18 | 12 | 3 | 3 | 0 | 73.7 | 505.8 | **15 %** |
| phi-3.5 exp17 / `dense` | no_change | +0.00 % | 24 | 0 | 20 | 4 | 0 | 0.0 | 690.2 | 0 % |
| phi-3.5 onA / `dense` | shipped | -7.58 % | 18 | 15 | 1 | 2 | 0 | 78.6 | 187.8 | **42 %** |
| qwen3-27B B / `full_attention` | no_change | +0.00 % | 20 | 0 | 8 | 10 | 2 | 0.0 | 618.4 | 0 % |
| qwen3-27B B / `linear_attention` | no_change | +0.00 % | 4 | 0 | 0 | 4 | 0 | 0.0 | 429.2 | 0 % |
| qwen3-27B FN / `full_attention` | improved | -2.25 % | 23 | 1 | 4 | 14 | 4 | 7.8 | 178.8 | **4 %** |

**17 of 26 (cell, layer-kind) pairs kept zero chains.**

## The part that answers the question properly: buffer type vs geometry

`kept` counts chains, not compliance. What each of the **9 cells that changed anything** actually shipped:

| cell | Δ /layer | what shipped | the advised **geometry**? |
|---|---|---|---|
| **gemma-4-26B onA** | -12.98 % | `88-core width-sharded hidden-width RMSNorm`. The advisor advised `l1/width_sharded/1x88` = 88 cores on seven 44 µs `rms_norm` rows. **Same grid.** | yes — **verbatim** |
| **north-mini FN** | -9.26 % | `32-core L1-width-sharded RMSNorm` on both MoE kinds. The advisor advised **22** — and here 22 is correct, the bounding box and the grid product agree. The cell measured `advisor_moe_norm_22` (0.5432) against its own 32 (0.5184) and 64 (0.5733). | strategy yes, grid no — **and the deviation is proved** |
| **phi-3.5 onA** | -7.58 % | `rope_storage: "L1 interleaved"`, `restore_geometry: "32-core exact rectangular height shard"`. **The advisor advised 32** — the 22 in its reconciliation is the truncated bounding box. | **yes** — it matched the advice and filed it as an override |
| **phi-3.5 B** | -5.09 % | Its rope chain runs `multiply`/`add` in the query's own memory config = the advised 32-core height shard; `neg`/`concat` stay interleaved. (`decode_core_grid: [8,1]` is the pre-existing residual-stream grid, not part of the change.) | **partly yes** — 32 is what the advisor advised, despite its `rejected_knobs` calling 11/22 "not recommendations" |
| **phi-3.5 FN** | -4.91 % | `advisor_rope_l1: "query_key"` — L1 interleaved throughout. The advised `height_sharded/32x1` **runs, is bit-identical, and is −10.43 %**; it was never tried. | no — buffer type only, **unproven** |
| **gemma-4-12B exp11** | -1.83 % | `keep_q_l1…`, `keep_k_l1…`, `keep_v_l1…`, `mlp_direct_down_input_layout`, `o_chain_l1` — all L1 **residency**. No grid. | no — buffer type |
| **qwen3-27B FN** | -2.25 % | "one L1-interleaved conversion before four slices". | no — buffer type |
| **gemma-4-26B FN** | -1.69 % | "keep `nlp_concat_heads_decode` output sharded into output projection; retain SDPA DRAM boundary". **The 88-core norm was tried in this arm and regressed.** | no — boundary decision |
| **gemma-4-26B B** | -0.47 % | `dram_sharded_roles: [o_proj, packed_mlp_gate_up, mlp_down]`. Its `rejected_knobs` states the *"advisor norm/grid direction was not booked into the shipped attention-boundary win"*. | no — DS roles |

**Exactly one cell in the corpus shipped the advisor's advised grid — and it produced the corpus's
largest win, −12.98 %.** Three more shipped a *self-chosen* grid on the op class the advisor pointed at,
and the two cells that measured the advisor's number head-to-head against their own both found the
advisor's worse. The remaining five shipped buffer-type or boundary changes with no grid at all.

| what shipped | cells |
|---|---|
| **the advised sharding strategy *and* core count** | **3** — gemma-4-26B onA (`width_sharded` 88), phi-3.5 onA (`height_sharded` 32), phi-3.5 B (32, on the 96-wide ops) |
| the advised strategy, a **self-chosen grid, proved better** | **1** — north-mini FN (32 vs advised 22, measured: 0.5184 vs 0.5432) |
| buffer type / boundary only, **no grid** | **5** — gemma-4-12B, gemma-4-26B FN, gemma-4-26B B, phi-3.5 FN, qwen3-27B FN |

**Corrected from the first version of this file, which said "one".** phi onA and phi B both shipped 32-core
height shards; the advisor advised **32** (`32x1`, `core_ranges` 22 + 10), not the 22 their reconciliations
reported. Both filed it as their own choice — phi B's `rejected_knobs` literally says *"advisor core counts
11/22 alone (not recommendations)"*.

So the honest one-line answer to *"how much was the advice followed?"*: **the advisor's L1-vs-DRAM call was
followed widely; its sharding in 3 of the 9 cells that changed anything, and its exact grid in those same 3.**
Of the 5 that dropped the sharding entirely, **4 never tried it** — and in the one cell I could measure, the
advised form is **−10.43 % against the −4.88 % that shipped**, bit-identical, so the deviation cost
≈1.43 ms/model ([`ADVICE-FAITHFULNESS`](ADVCHAL-V2-ADVICE-FAITHFULNESS.md) §4).

## Two of the three biggest wins came from cells told not to bother

| cell | Δ | feasibility verdict | `ceiling_us` | floor µs | its chains |
|---|---|---|---|---|---|
| gemma-4-26B onA / `full_attention` | **-12.98 %** | **not_measurable** | **0** | 3.543 | 4× below_threshold |
| gemma-4-26B onA / `sliding_attention` | **-12.98 %** | **not_measurable** | **0** | 0.587 | 3× below_threshold |
| north-mini FN / `dense_full_attention` | **-9.26 %** | **measurable** | **3.476** | 0.849 | 1× not_measurable, 7× below_threshold |
| north-mini FN / `full_attention_moe` | **-9.26 %** | **not_measurable** | **0** | 1.206 | 1× not_measurable |
| north-mini FN / `sliding_attention_moe` | **-9.26 %** | **not_measurable** | **1.148** | 1.638 | 3× not_measurable |

Both shipped a widened RMSNorm. Both had layer kinds whose feasibility arithmetic said `ceiling_us = 0` —
*"the advisor proposes removing zero microseconds"* — verdict `not_measurable`, every chain
`below_threshold` or `not_measurable`. **Both recorded 0 kept chains.** Both screened anyway and shipped
−12.98 % and −9.26 % per layer.

**The ceiling prices boundary conversions only.** An in-chain re-grid — which is exactly what a 1→88 or
1→32-core RMSNorm is — is worth `0.000 µs` to it. So the metric that gates screening cannot see the two
largest wins in the corpus, by construction, and the chain bookkeeping records them as not followed.
→ [`ADVISOR-VALUE`](ADVCHAL-V2-ADVISOR-VALUE.md) §4, [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) §D0.

## How much was dismissed without measurement, checked against each cell's own floor

134 chains were dismissed as `below_threshold` or `not_measurable`. Comparing each chain's op cost
against **its own cell's** noise floor, **70 of them are ≥ 5× the floor**:

| cell / kind | chain | verdict | chain µs | floor µs | × floor | % of window | ops |
|---|---|---|---|---|---|---|---|
| gemma-4-26B onA / `sliding_attention` | `sliding_attention:0` | `below_threshold` | 165.8 | 0.587 | **282×** | 9.3 % | `rms_norm`, `linear` |
| qwen3-27B B / `full_attention` | `full_attention:16` | `below_threshold` | 183.2 | 0.749 | **245×** | 14.5 % | `linear`, `add`, `rms_norm`, `multiply` … |
| gemma-4-26B onA / `sliding_attention` | `sliding_attention:2` | `below_threshold` | 143.5 | 0.587 | **244×** | 8.0 % | `rms_norm`, `add`, `rms_norm`, `slice_static` … |
| phi-3.5 exp17 / `dense` | `dense:14` | `below_threshold` | 196.2 | 1.092 | **180×** | 19.3 % | `add`, `rms_norm`, `linear`, `slice_static` … |
| phi-3.5 exp17 / `dense` | `dense:15` | `below_threshold` | 185.4 | 1.092 | **170×** | 18.2 % | `linear`, `add` |
| gemma-4-26B B / `full_attention` | `full_attention:0` | `below_threshold` | 133.6 | 0.852 | **157×** | 11.0 % | `rms_norm`, `linear` |
| gemma-4-26B B / `sliding_attention` | `sliding_attention:0` | `below_threshold` | 167.8 | 1.291 | **130×** | 13.9 % | `rms_norm`, `linear` |
| llama-3.2-1B exp17 / `dense` | `dense:1` | `below_threshold` | 17.4 | 0.146 | **119×** | 4.9 % | `nlp_create_qkv_heads_decode`, `rotary_embedding_llama` |
| gemma-4-26B B / `full_attention` | `full_attention:3` | `below_threshold` | 90.1 | 0.852 | **106×** | 7.4 % | `rms_norm`, `add`, `rms_norm` |
| gemma-4-26B FN / `full_attention` | `full_attention:0` | `below_threshold` | 187.9 | 2.157 | **87×** | 12.9 % | `rms_norm`, `linear` |
| north-mini onA / `full_attention_sparse_moe` | `full_attention_sparse_moe:0` | `not_measurable` | 69.9 | 0.847 | **83×** | 8.5 % | `rms_norm`, `linear` |
| gemma-4-26B B / `sliding_attention` | `sliding_attention:2` | `below_threshold` | 90.1 | 1.291 | **70×** | 7.5 % | `rms_norm`, `add`, `rms_norm` |
| north-mini onA / `dense_full_attention` | `dense_full_attention:2` | `not_measurable` | 118.6 | 1.841 | **64×** | 43.2 % | `linear`, `slice_static`, `slice_static`, `silu` … |
| north-mini FN / `full_attention_moe` | `full_attention_moe:0` | `not_measurable` | 70.0 | 1.206 | **58×** | 14.1 % | `rms_norm`, `linear` |
| gemma-4-26B onA / `full_attention` | `full_attention:0` | `below_threshold` | 202.9 | 3.543 | **57×** | 10.2 % | `rms_norm`, `linear` |
| gemma-4-26B FN / `sliding_attention` | `sliding_attention:0` | `below_threshold` | 156.9 | 2.804 | **56×** | 12.4 % | `rms_norm`, `linear` |
| phi-3.5 onA / `dense` | `dense:0` | `below_threshold` | 22.6 | 0.43 | **53×** | 4.0 % | `rms_norm`, `nlp_create_qkv_heads_decode` |
| gemma-4-26B B / `full_attention` | `full_attention:5` | `below_threshold` | 44.3 | 0.852 | **52×** | 3.7 % | `rms_norm` |
| phi-3.5 onA / `dense` | `dense:12` | `below_threshold` | 21.6 | 0.43 | **50×** | 3.8 % | `add`, `rms_norm`, `multiply`, `add` |
| north-mini FN / `sliding_attention_moe` | `sliding_attention_moe:0` | `not_measurable` | 70.1 | 1.638 | **43×** | 13.4 % | `rms_norm`, `linear` |
| gemma-4-26B onA / `full_attention` | `full_attention:3` | `below_threshold` | 142.8 | 3.543 | **40×** | 7.2 % | `rms_norm`, `add`, `rms_norm`, `slice_static` … |
| phi-3.5 B / `dense` | `dense:0` | `below_threshold` | 27.5 | 0.713 | **39×** | 3.9 % | `rms_norm`, `nlp_create_qkv_heads_decode` |

*(top 22 of 70; full list in `advchal-v2-followed.json`)*

**Caveat, stated because it matters:** a chain's µs is the *incumbent cost of its ops*, not the saving the
advisor claims. So "57× the floor" does not mean a 57× win is available — it means **a saving of 1.7 % of
those ops would already be measurable**. That is the sense in which the dismissal is not justified by floor
arithmetic. The empirical check is the section above: the cell whose ceiling said `0` shipped −12.98 %.

## What cannot be checked, and why

**Op-level verification — did the shipped placement match the advised one, per op — is possible for exactly
one cell.** Only phi-3.5 FN saved a before/after profile pair (`profiles/dense_incumbent.csv` +
`dense_winner.csv`). Every other cell saved either incumbent-side profiles only or none at all, so for them
"was the advice followed?" can be answered at chain-verdict level (above) but **not** at op-placement level.
→ [`PERF-REPORT-AUDIT`](ADVCHAL-V2-PERF-REPORT-AUDIT.md), [`IMPROVEMENTS`](ADVCHAL-V2-IMPROVEMENTS.md) §B0.

**And 21.8 % of the op↔advice pairings are positional guesses corpus-wide** (111 of 509), the tool's own documented limitation. Per cell it ranges from 0 % to 38 %. → §C5e.

## Bucket shares, corpus-wide

| bucket | ops | µs | share of profiled op time |
|---|---|---|---|
| `untraced` | 554 | 18630 | **48.3 %** |
| `chain` | 334 | 5547 | 13.8 % |
| `dram_resident` | 86 | 1182 | 3.1 % |

**Nearly half of all profiled op time is `untraced` — invisible to the advisor.** The advice-followed
question applies to the 17 % the advisor can see and disagrees about. That framing matters more than any
number above it.

---

Generated from the cells' own committed artefacts by `audit_followed.py`. Raw data:
`advchal-v2-followed.json`.

**Two provenance notes, found while assembling this:**

1. **Two cells' stage output never landed on a `cell/advchal-v2/*` branch.** gemma-4-26B B lives on
   `run/advchal-v2/nofuse-noadvise/google_gemma_4_26b_a4b_it` and north-mini FN on
   `wip/advchal-v2/fuse-noadvise/coherelabs_north_mini_code_1_0`. Both have complete artefacts; neither is
   tagged like the other 13. This is the outstanding tagging item — it needs a `git tag -a` I cannot make.
2. **`advchal-v2-data.json` has 14 rows, not 15** — it is missing **gemma-4-26B FN**. Any count derived from
   that file alone is one cell short. This file is built from the branches, so it has all 15.
