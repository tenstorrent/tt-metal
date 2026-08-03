# advchal-v2 — per cell: what actually happened to the advice

**14 cells, 24 layer-kind runs.** Generated from each cell's own `reconciliation_<kind>.json` and `final.json` — tags where they exist, the cell's work branch where the optimization is complete but the tag was withheld. The question here is not *how much did it gain* but **why**: was the advice already shipped, tested and refuted, out of reach, or right.

| column | meaning |
|---|---|
| **already shipped** | `agrees_with_shipped` — the advisor, which never sees your layouts, independently re-derived what already ships. Zero marginal gain **by construction**, not wasted advice |
| **differs** | `chain` — placement the advisor would change |
| **DRAM** | `dram_resident` — the advisor wants it de-sharded; that is advice too |
| **unreachable** | `untraced` — in the profile, absent from the advisor's graph (terminal ops) |
| **ceiling** | µs of shipped conversions the advice does *not* place: the most attributable to the advisor |
| **boundary µs** | channel 1 — `us_advisor_drops`: conversions the advice does not place. Prices only what a chain-lengthening change *removes* |
| **re-grid µs** | channel 2 — device time of ops the advisor puts on a **different grid** than shipped. A re-grid inside a chain removes no conversion, so channel 1 prices it at **exactly zero** |
| **of which starved** | the subset on **<=2 cores**. The highest-yield subset, and the one channel 1 is blindest to |
| **worse** | chains screened and rejected **because they measured slower than the incumbent** — the bad-advice count, measured rather than assumed |

| cell | state | kind | already shipped | differs | unreachable | **boundary µs** | **re-grid µs** | **of which starved** | kept | **worse** |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma-4-12B | tagged | `full_attention` | 58.7 % | 27.0 % | 2.1 % | 25.39 | 253.0 | **15.8** | 8 | **0** |
| gemma-4-12B | tagged | `sliding_attention` | 61.7 % | 19.2 % | 4.8 % | 15.782 | 179.5 | **0.0** | 6 | **0** |
| phi-3.5 exp17 | tagged | `dense` | 14.4 % | 49.2 % | 9.1 % | 83.551 | 472.0 | **0.0** | 0 | **16** |
| llama-3.1-8B | tagged | `dense` | 76.3 % | 7.8 % | 4.5 % | 4.394 | 50.8 | **0.0** | 0 | **2** |
| llama-3.2-1B | tagged | `dense` | 72.8 % | 11.4 % | 7.4 % | 2.822 | 40.4 | **0.0** | 0 | **1** |
| phi arm FN | tagged | `dense` | 6.7 % | 62.1 % | 13.4 % | 71.637 | 422.9 | **88.8** | 12 | **0** |
| phi arm B | tagged | `dense` | 30.7 % | 19.3 % | 10.7 % | 70.732 | 107.2 | **0.0** | 12 | **0** |
| phi arm onA | tagged | `dense` | 39.4 % | 21.6 % | 14.8 % | 70.381 | 101.5 | **0.0** | 15 | **0** |
| qwen fuse-noadvise | tagged | `full_attention` | 69.2 % | 12.3 % | 4.5 % | 34.282 | 101.6 | **0.0** | 1 | **4** |
| qwen nofuse-noadvise | tagged | `full_attention` | 42.6 % | 45.0 % | 2.5 % | 33.698 | 547.3 | **0.0** | 0 | **8** |
| qwen nofuse-noadvise | tagged | `linear_attention` | 2.9 % | 2.7 % | 63.5 % | 0 | 429.2 | **0.0** | 0 | **0** |
| gemma-4-26B | **complete**, untagged | `full_attention` | 12.7 % | 27.4 % | 52.1 % | 6.373 | 329.6 | **203.1** | 0 | **5** |
| gemma-4-26B | **complete**, untagged | `sliding_attention` | 13.5 % | 27.0 % | 52.4 % | 3.832 | 323.4 | **193.9** | 2 | **0** |
| north-mini FN | **complete**, untagged | `dense_full_attention` | 46.9 % | 8.0 % | 38.8 % | 3.476 | 8.3 | **0.0** | 0 | **0** |
| north-mini FN | **complete**, untagged | `full_attention_moe` | 5.1 % | 14.1 % | 67.9 % | 0 | 70.0 | **26.1** | 0 | **0** |
| north-mini FN | **complete**, untagged | `sliding_attention_moe` | 4.9 % | 13.4 % | 69.0 % | 1.148 | 70.1 | **26.1** | 0 | **0** |
| north-mini B | tagged | `dense_full_forced_rope` | 40.6 % | 39.7 % | 6.7 % | 5.929 | 69.9 | **0.0** | 0 | **0** |
| north-mini B | tagged | `full_no_rope_moe` | 12.5 % | 1.6 % | 76.4 % | 0.563 | 7.8 | **0.0** | 0 | **0** |
| north-mini B | tagged | `sliding_rope_moe` | 12.4 % | 1.6 % | 75.7 % | 1.688 | 7.9 | **0.0** | 0 | **0** |
| gemma-4-26B onA | tagged | `full_attention` | 16.8 % | 20.5 % | 58.5 % | 0 | 402.1 | **203.4** | 0 | **0** |
| gemma-4-26B onA | tagged | `sliding_attention` | 12.5 % | 18.3 % | 64.7 % | 0 | 323.7 | **194.6** | 0 | **0** |
| north-mini onA | tagged | `dense_full_attention` | 9.3 % | 80.4 % | 4.5 % | 1.709 | 215.4 | **26.0** | 0 | **0** |
| north-mini onA | tagged | `full_attention_sparse_moe` | 3.2 % | 12.4 % | 77.2 % | 0.562 | 100.4 | **26.1** | 0 | **0** |
| north-mini onA | tagged | `sliding_attention_sparse_moe` | 3.1 % | 12.3 % | 76.6 % | 1.706 | 100.4 | **26.0** | 0 | **0** |

Op-by-op detail per model and kind — including what was rejected as slower and what could have been tried and was not — is in [`ADVCHAL-V2-PER-OP.md`](ADVCHAL-V2-PER-OP.md).

## Classification — derived from the numbers above

**gemma-4-12B** — `improved`, 2 optimization round(s)
- `full_attention` → **advice won**: 8 chains kept, **0 measured worse**, from a ceiling of 25.39 µs (7.51× floor) with only 58.7 % already shipped
- `sliding_attention` → **advice won**: 6 chains kept, **0 measured worse**, from a ceiling of 15.782 µs (22.17× floor) with only 61.7 % already shipped

**phi-3.5 exp17** — `no_change`, 1 optimization round(s)
- `dense` → **tested and refuted**: ceiling 83.551 µs at 76.51× the floor, and **16 chains measured slower** than the incumbent

**llama-3.1-8B** — `contribution_zero`, 3 optimization round(s)
- `dense` → **already shipped**: 76.3 % agreement and a ceiling of only 4.394 µs; of what differed, 2 measured worse and 2 were below threshold

**llama-3.2-1B** — `no_change`, 3 optimization round(s)
- `dense` → **already shipped**: 72.8 % agreement and a ceiling of only 2.822 µs; of what differed, 1 measured worse and 2 were below threshold

**phi arm FN** — `improved`, 1 optimization round(s)
- `dense` → **advice won**: 12 chains kept, **0 measured worse**, from a ceiling of 71.637 µs (67.33× floor) with only 6.7 % already shipped

**phi arm B** — `improved`, 2 optimization round(s)
- `dense` → **advice won**: 12 chains kept, **0 measured worse**, from a ceiling of 70.732 µs (99.2× floor) with only 30.7 % already shipped

**phi arm onA** — `shipped`, 1 optimization round(s)
- `dense` → **advice won**: 15 chains kept, **0 measured worse**, from a ceiling of 70.381 µs (163.68× floor) with only 39.4 % already shipped

**qwen fuse-noadvise** — `improved`, 3 optimization round(s)
- `full_attention` → **tested and refuted**: ceiling 34.282 µs at 21.31× the floor, and **4 chains measured slower** than the incumbent  ⚠ warm-up ramp

**qwen nofuse-noadvise** — `no_change`, 3 optimization round(s)
- `full_attention` → **tested and refuted**: ceiling 33.698 µs at 44.99× the floor, and **8 chains measured slower** than the incumbent
- `linear_attention` → **nothing attributable — the advisor agrees**: all 4886.983 µs of boundary time is endorsed by the advice (3988.953 µs over 13 edges); 63.5 % of the window is also unreachable. Real cost, but not this stage's to claim  ⚠ warm-up ramp

**gemma-4-26B** — `SHIPPED`, 1 optimization round(s)  ·  *complete, untagged*
- `full_attention` → **out of reach**: 52.1 % of the window is invisible to the tracer; ceiling only 6.373 µs
- `sliding_attention` → **out of reach**: 52.4 % of the window is invisible to the tracer; ceiling only 3.832 µs

**north-mini FN** — `improved`, 1 optimization round(s)  ·  *complete, untagged*
- `dense_full_attention` → **inconclusive**: ceiling 3.476 µs, 0 kept, 0 worse, 7 below threshold
- `full_attention_moe` → **zero is not evidence — nothing was comparable**: 11 boundaries undetermined (23.952 µs), 3 unresolved (40.192 µs), **0 µs agreed**. At 67.9 % untraced the advised-op adjacency keeps breaking, so the comparison was never askable
- `sliding_attention_moe` → **out of reach**: 69.0 % of the window is invisible to the tracer; ceiling only 1.148 µs

**north-mini B** — `no_change`, 3 optimization round(s)
- `dense_full_forced_rope` → **inconclusive**: ceiling 5.929 µs, 0 kept, 0 worse, 2 below threshold
- `full_no_rope_moe` → **out of reach**: 76.4 % of the window is invisible to the tracer; ceiling only 0.563 µs  ⚠ warm-up ramp
- `sliding_rope_moe` → **out of reach**: 75.7 % of the window is invisible to the tracer; ceiling only 1.688 µs

**gemma-4-26B onA** — `improved`, 2 optimization round(s)
- `full_attention` → **zero is not evidence — nothing was comparable**: 5 boundaries undetermined (12.711 µs), 2 unresolved (8.397 µs), **8.826 µs agreed**. At 58.5 % untraced the advised-op adjacency keeps breaking, so the comparison was never askable
- `sliding_attention` → **zero is not evidence — nothing was comparable**: 5 boundaries undetermined (12.831 µs), 2 unresolved (8.808 µs), **7.882 µs agreed**. At 64.7 % untraced the advised-op adjacency keeps breaking, so the comparison was never askable

**north-mini onA** — `no_change`, 1 optimization round(s)
- `dense_full_attention` → **inconclusive**: ceiling 1.709 µs, 0 kept, 0 worse, 0 below threshold
- `full_attention_sparse_moe` → **out of reach**: 77.2 % of the window is invisible to the tracer; ceiling only 0.562 µs
- `sliding_attention_sparse_moe` → **out of reach**: 76.6 % of the window is invisible to the tracer; ceiling only 1.706 µs

## Two attribution channels, validated against what was realised

A single channel under-reports. Realised saving per layer, against each channel's upper bound:

| cell | kind | boundary µs | starved re-grid µs | realised/layer | explained by |
|---|---|---|---|---|---|
| gemma-4-12B | `full_attention` | 25.39 | 15.8 | 49.9 | **needs channel 2** |
| gemma-4-12B | `sliding_attention` | 15.782 | 0.0 | 22.7 | **exceeds channel 1** |
| phi-3.5 exp17 | `dense` | 83.551 | 0.0 | 0.0 | nothing realised |
| llama-3.1-8B | `dense` | 4.394 | 0.0 | 0.0 | nothing realised |
| llama-3.2-1B | `dense` | 2.822 | 0.0 | 0.0 | nothing realised |
| phi arm FN | `dense` | 71.637 | 88.8 | 39.6 | channel 1 alone |
| phi arm B | `dense` | 70.732 | 0.0 | 40.2 | channel 1 alone |
| phi arm onA | `dense` | 70.381 | 0.0 | 49.8 | channel 1 alone |
| qwen fuse-noadvise | `full_attention` | 34.282 | 0.0 | 27.1 | channel 1 alone |
| qwen nofuse-noadvise | `full_attention` | 33.698 | 0.0 | 0.0 | nothing realised |
| qwen nofuse-noadvise | `linear_attention` | 0 | 0.0 | 0.0 | nothing realised |
| gemma-4-26B | `full_attention` | 6.373 | 203.1 | 0.0 | nothing realised |
| gemma-4-26B | `sliding_attention` | 3.832 | 193.9 | 5.9 | **needs channel 2** |
| north-mini FN | `dense_full_attention` | 3.476 | 0.0 | 0.0 | nothing realised |
| north-mini FN | `full_attention_moe` | 0 | 26.1 | 33.5 | **needs channel 2** |
| north-mini FN | `sliding_attention_moe` | 1.148 | 26.1 | 59.7 | **needs channel 2** |
| north-mini B | `dense_full_forced_rope` | 5.929 | 0.0 | 0.0 | nothing realised |
| north-mini B | `full_no_rope_moe` | 0.563 | 0.0 | 0.0 | nothing realised |
| north-mini B | `sliding_rope_moe` | 1.688 | 0.0 | 0.0 | nothing realised |
| gemma-4-26B onA | `full_attention` | 0 | 203.4 | 237.3 | **needs channel 2** |
| gemma-4-26B onA | `sliding_attention` | 0 | 194.6 | 236.8 | **needs channel 2** |
| north-mini onA | `dense_full_attention` | 1.709 | 26.0 | 0.0 | nothing realised |
| north-mini onA | `full_attention_sparse_moe` | 0.562 | 26.1 | 0.0 | nothing realised |
| north-mini onA | `sliding_attention_sparse_moe` | 1.706 | 26.0 | 0.0 | nothing realised |

**Channel 1 under-predicts four of the kinds that gained.** north-mini `sliding_attention_moe` realised
**59.7 us/layer against a boundary ceiling of 1.148 us** -- 52x -- and `full_attention_moe` realised 33.5 us
against a ceiling of **exactly 0**. gemma-4-12B realised 49.9 and 22.7 us/layer against ceilings of 25.4 and
15.8. So the boundary channel is not merely incomplete for the MoE cells; it under-states gemma-12B too, which
means part of the corpus's headline win was also a re-grid and was never attributed as one.

## Channel 2 must be split by direction, or it credits known-bad advice

The re-grid total is **not** headroom. Most of it is the advisor advising *fewer* cores than shipped --
`linear 86->77`, `103->99`, `96->88`, `nlp_create_qkv_heads_decode 32->22` -- which is the systematic bias the
v1 audit measured at **8 of 8 non-matmul ops**, and which phi-3.5 exp17 refuted **16 times** in one cell. Only
the **UP** direction is a candidate:

| direction | meaning | trust |
|---|---|---|
| **UP** — advised cores > shipped, especially shipped <=2 | a starved op the advisor wants sharded | **the real signal** |
| **DOWN** — advised cores < shipped | the fewer-cores bias; measured slower wherever it was screened | not headroom |

**The highest-yield finding class in the whole corpus is a 1-core RMSNorm the advisor wants sharded**, and it
occurs in two different models:

| cell | kind | op | shipped | advised | op time |
|---|---|---|---|---|---|
| north-mini FN | `full_attention_moe` | `rms_norm` | **1 core** | 22 | 26.1 us |
| north-mini FN | `sliding_attention_moe` | `rms_norm` | **1 core** | 22 | 26.1 us |
| gemma-4-26B | `sliding_attention` | `rms_norm` | **1 core** | 88 | 44.5 us |
| gemma-4-26B | `sliding_attention` | `rms_norm` | **1 core** | 88 | 44.1 us |

north-mini turned its 26.1 us into a **20.5 us saving per layer (4.65x on the op)** by sweeping 22/32/64 and
taking 32. gemma-4-26B has ~88.6 us of the same shape and realised **5.9 us/layer**. Why the two differ is a
concrete, checkable lead -- not a claim that value was left on the table.

**Correcting an earlier reading of mine:** I previously wrote that gemma-4-26B "left 203 us of starved re-grid
headroom untouched". That was wrong on two counts. The 203 us figure summed rows in both directions, and its
kept chains do include a 1-core `rms_norm`, so it was not ignored. What is true and narrower: its 1-core norms
total ~88.6 us and it realised 5.9 us/layer where a sibling model realised 78 % of the equivalent op's time.

## A zero ceiling has two meanings, and they are opposite

`boundary µs` = 0 renders identically in the verdict but arises two ways:

| condition | reading |
|---|---|
| 0 with a large **agrees** | the advisor places a conversion on every edge the shipped graph does. Genuine *nothing attributable* — real cost, belongs to `$optimize` |
| 0 with **undetermined + unresolved** dominant | **no comparison was possible.** The ceiling is not evidence of absence |

qwen `linear_attention` is the first: 4886.983 µs of boundary time, **3988.953 µs agreed across 13 edges**,
almost all `retilize` — the dearest conversion class there is, and the advice would keep all of it.

north-mini `full_attention_moe` is the second: 64.144 µs of boundary time, **0 agreed**, 11 boundaries
`undetermined` and 3 `unresolved`. Not one of its 14 boundary ops was comparable. The mechanism is adjacency —
a boundary falls to `undetermined` when no advised op is paired beside it, and at **68 % untraced** the pairing
keeps breaking. So a high unreachable share structurally drives the ceiling toward zero, which is also why
gemma-4-26B reads 6.4 and 3.8 µs at 52 % untraced.

That matters because north-mini realised **33.5 µs/layer** on that same kind, from a re-grid. A flavour-2 zero
carries no information about opportunity at all.

## The pattern

Four distinct causes sit behind the single word in the `outcome` field, and only one of them is a verdict on
the advisor:

1. **Already shipped** — high agreement, tiny ceiling. The advisor re-derived a decoder it had never seen.
   Nothing to gain, and a *positive* statement about its judgement.
2. **Advice won** — low agreement, large ceiling, chains kept and none slower.
3. **Tested and refuted** — low agreement, large ceiling, and candidates that measured *worse*. A real
   negative, and only readable because the floor was tight enough to resolve it.
4. **Out of reach / nothing attributable** — the tracer could not see it, or the advisor endorses every
   conversion already present. Not a verdict on the advisor at all.

The clean split is **agreement × reachability**, not model size or layer kind.

