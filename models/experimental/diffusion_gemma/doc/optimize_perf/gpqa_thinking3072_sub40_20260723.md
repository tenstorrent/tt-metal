# GPQA-Diamond thinking @3072 — 40-question subset (2026-07-23)

Status: provenance-only — run on `DG_VLLM_GUMBEL_MODE=host`, deleted 2026-07-28, and before the
concat MoE became the only denoise MoE, so the absolute score is not a current result.
Owns: the outcome-split of a thinking-mode run and the context-length bottleneck it exposed.
See also: [refuted list](../REFUTED.md), [optimize-perf hub](README.md).

vLLM end-to-end, up-front capture, thinking mode (`<|think|>`), `MAX_GEN_TOKS=3072` (12 blocks),
`MAX_MODEL_LEN=4096`, first 40 GPQA-Diamond questions. Overall `exact_match = 0.35 ± 0.076` (14/40).

| outcome | n | acc |
|--|--:|--:|
| FINISHED (reasoning completed, has answer) | 20 (50%) | 14/20 = 0.70 |
| TRUNCATED (hit 3072, no answer) | 15 (37.5%) | 0/15 = 0.00 |
| DEGENERATE (repetition) | 5 (12.5%) | 0/5 = 0.00 |

**MEASUREMENT TRAP — three denominators.** The 0.35 aggregate hides that thinking scores 0.70 when
it *finishes*; reporting the aggregate alone would hide the effect entirely. The full
three-denominator rule (extractable / non-empty / all) and the other GPQA traps live in
[decision fidelity](../decision_fidelity/README.md).

20/40 (50%) of traces hit the 12-block (3072-token) cap, so the binding constraint was **context
length, not the sampler**. Inside `MAX_MODEL_LEN=4096` the generation budget can only reach ~3584
tokens (14 blocks); realizing thinking's benefit needs `MAX_MODEL_LEN` above 4096, which costs
O(`p_max`) reveal reads per step plus a larger KV.

The in-run comparison line "Reference (full 198, different sample set): argmax 61.11%, non-thinking
host-Gumbel 56.57%" is neither sample-matched nor generation-budget-matched to this 40-question
@3072 result and must not be read as a delta; a reference score is comparable only at a matched
generation budget. The current full-198 GPQA number lives in
[decision fidelity](../decision_fidelity/README.md).

Artifact: `gpqa_thinking3072_sub40_20260723.samples.jsonl.gz`.
