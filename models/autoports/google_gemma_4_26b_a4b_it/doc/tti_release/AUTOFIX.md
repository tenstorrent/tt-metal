# AutoFix Report

## Starting evidence

- The first no-Docker CI-nightly release wrote valid lm-eval JSON but reported
  `no eval results parsed (rc=0)` for `meta_ifeval` and `meta_gpqa_cot`.
- The same run skipped spec tests because Gemma 4 had no TTI server-test mapping.
- GPQA scored 0/10; eight responses ended mid-reasoning with `[invalid]` extraction.

## Hypothesis experiments

- **Release-row alias mismatch — verified.** TTI invoked public lm-eval tasks
  `ifeval` and `gpqa_diamond_cot_zeroshot`, while its parser searched only for
  stable report names `meta_ifeval` and `meta_gpqa_cot`. Alias-aware result
  matching now scores the public keys while preserving stable report-row names.
- **GPQA generation truncation — verified.** The aggregate config omitted
  `max_gen_toks`; all sampled responses were short and most stopped before the
  requested boxed answer. The task now uses a 32,768-token reasoning budget,
  still within the unchanged 262,144-token context contract.
- **Missing spec-test mapping — verified.** The filter had no Gemma 4 model or
  generic LLM matrix entry. The p300x2 mapping now selects
  `VLLMParamConformanceTest`.

## Verification

- Focused TTI tests: 40 passed.
- TTI fixes are committed on `codex/gemma4-stage11`; the live release rerun is
  the final integration proof.

## Final status

Harness defects are fixed locally. No device reset or model implementation
change was needed. Live GPQA, parser, benchmark, and API-conformance evidence
must pass before Stage 11 closes.
