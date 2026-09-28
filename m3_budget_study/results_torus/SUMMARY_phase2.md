# M3 prefill MoE on the full 8x4 BH galaxy — dispatch v2 + combine v2 (bf8), phase 2 (T2)

Branch `vmelnykov/m3_moe_fabric2d` @ 191eeb2: main + PRs #57859/#57654 (`dispatch_fabric2d`) + local BF8-input
support for `combine_fabric2d` + M3 knobs. Whole 8x4 (SP=8, TP=4, EP=32), bf4 experts, untraced, reset per process.
No hangs, no timing failures. Data: `runs.csv` rows `t2_*`, `logs/`, `profiles/*_t2_*_c8192_bf4/zones.{txt,json}`.

Configs: A = 1d, all Linear, v1/v1 (today). R = torus, CCL Ring, MoE Ring, v1/v1. E1 = R + combine v2 (bf8 in).
E1c = E1 with the old bf16 typecast. E2 = R + dispatch v2. E3 = R + dispatch v2 + combine v2.

**Step 0** — `test_ttnn_combine_fabric2d` torus-xy-8x4: 6/6 pass (row_major, tile, tile_bfp8 × pcc, perf_no_pcc).

## Wall, S8 (layers 8–15), h = 0, B = 1, median of 5 (ms)

| cfg | W = 8192 | W = 16384 |
|---|---|---|
| A | 106.20 | 196.60 |
| R | 110.56 | 206.09 |
| E1 | 97.17 | 184.04 |
| E1c | 112.77 | – |
| E2 | 97.97 | 181.06 |
| **E3** | **91.04 (−14.3%)** | **171.25 (−12.9%)** |

## Zones, sparse layer 3, W = 8192, worst chip, device ms

| zone | A | R | E1 | E2 | E3 |
|---|---|---|---|---|---|
| dispatch | 2.59 | 2.87 | 2.77 | 1.59 | 1.59 |
| experts_mm | 2.93 | 2.94 | 2.94 | 2.93 | 2.93 |
| combine | 5.87 | 6.53 | 4.38 | 5.28 | 4.13 |
| moe_reduce | 7.51 | 8.39 | 4.89 | 4.48 | 3.92 |
| mlp | 10.34 | 10.49 | 8.53 | 8.26 | 7.21 |
| layer | 14.53 | 14.99 | 13.06 | 12.76 | 11.64 |

73 ops per sparse layer in every config; no `combine_v2_prep` any more.
Host gap (profiled chunk wall / device / gap): A 25.3 / 15.73 / 9.6 (38%); R 26.6 / 16.03 / 10.6 (40%);
E1 43.5 / 14.25 / 29.2 (67%); E2 26.0 / 13.83 / 12.2 (47%); E3 37.0 / 12.71 / 24.3 (66%). With combine v2 the
op-to-op gaps roughly double in the profiled single-layer chunk; the untraced multi-layer sweeps are still faster.

## KV PCC (6 layers, longbook_5120; K / V / index_k)

| layer | A | E3 |
|---|---|---|
| L3 | .99972 / .99925 / .99975 | .99972 / .99925 / .99975 |
| L4 | .99900 / .99747 / .99927 | .99895 / .99736 / .99924 |
| L5 | .99902 / .99579 / .99933 | .99894 / .99542 / .99929 |

## Expert load (W = 8192, top-4 of 128, EP = 32 → 4 experts per chip)

per-chip max/mean, actual / best static packing (LPT; always equals the bound max(mean, hottest expert)/mean):

| layer | longbook | P1 (M3 .py code) | P2 (deepseek_prefill C++) |
|---|---|---|---|
| 8 | 4.37 / 3.25 | 6.65 / 6.65 | 6.02 / 5.48 |
| 9 | 7.85 / 5.54 | 7.20 / 4.10 | 7.00 / 5.02 |
| 10 | 6.77 / 3.77 | 3.55 / 3.01 | 3.52 / 3.01 |
| 11 | 6.99 / 6.86 | 6.82 / 6.81 | 7.12 / 6.93 |
| 12 | 7.86 / 7.83 | 7.79 / 7.76 | 7.82 / 7.80 |
| 13 | 6.51 / 6.40 | 6.73 / 6.57 | 6.34 / 6.15 |
| 14 | 7.82 / 7.47 | 8.42 / 7.56 | 8.65 / 7.36 |
| 15 | 5.13 / 5.01 | 6.15 / 6.14 | 6.20 / 6.18 |

per-column max/mean 1.19–2.15. The hottest expert takes 9–25% of token-expert pairs (up to 98% of tokens: L12
expert 109 gets 8022 of 8192) and is the same expert on every prompt (L12→109, L13→9, L14→42, L15→112); their
correction biases are ordinary, so it is structural in the gate weights, not prompt-driven. M3-tokenized
`longbook_56320`: mean actual 6.63, LPT 6.30.

**Trace caveat:** `longbook_qa_eng_prefill_56320_nopad` (the default trace of every earlier study harness) holds
DeepSeek-R1 token IDs, which decode to junk under the M3 tokenizer. Skew is similar on M3-tokenized text, but future
MoE measurements should use an M3-tokenized trace (`longbook_56320` or the code prompts).

## Interpretation

1. Both new ops cut MoE communication: dispatch −45% (2.87 → 1.59 ms); combine fed bf8 −33% (6.53 → 4.38 ms; the old
   typecast path was 2% slower than R); less waiting, so moe_reduce 8.39 → 3.92 ms in E3.
2. Net: E3 is 14.3% faster than A at 8192 (1.9 ms/layer), 12.9% at 16384; the device layer time drops 20%.
3. Ring CCL works at 8192 now; on its own it is still 4–5% slower than A — the gain comes with the v2 ops.
4. Skew dominates expert time: experts_mm is 2.93 ms on the worst chip vs 0.61 ms mean (0 on some chips), so ~79% of
   the worst chip's expert time is imbalance, and combine / moe_reduce mostly wait for that chip.
5. A static relabeling barely helps (5–13% off the per-layer max, ~0.3 ms/layer): one expert per layer sets the bound,
   so the lever is hot-expert replication or splitting a hot expert's tokens across chips.
