# GPQA-Diamond thinking@3072 — 40-sample subset (2026-07-23)

vLLM end-to-end, up-front-capture + early-halt + reveal-mask, host-Gumbel (official sampler),
thinking mode (`<|think|>`), MAX_GEN_TOKS=3072 (12 blocks), MAX_MODEL_LEN=4096. First 40 GPQA-Diamond.

Overall exact_match = 0.35 ± 0.076 (14/40).

Breakdown by response outcome:
| outcome | n | acc |
|--|--:|--:|
| FINISHED (reasoning completed, has answer) | 20 (50%) | 14/20 = 0.70 |
| TRUNCATED (hit 3072, no answer) | 15 (37.5%) | 0/15 = 0.00 |
| DEGENERATE (repetition) | 5 (12.5%) | 0/5 = 0.00 |

Block usage: 20/40 (50%) hit the 12-block (3072) cap.

Reference (full 198, different sample set): argmax 61.11%, non-thinking host-Gumbel 56.57%.

Conclusion: thinking helps WHEN it finishes (0.70 > argmax 0.61), but 50% of traces do not
finish within the 4096 context (37.5% truncated + 12.5% degenerate → 0), capping the aggregate
at 0.35. Bottleneck is context length (4096), not the sampler. Within 4096, MAX_GEN_TOKS can
only reach ~3584 (14 blocks) — marginal. Realizing thinking's benefit needs MAX_MODEL_LEN>4096
(slower: O(p_max) reveal reads + larger KV).
