# Exact CI-subset Hugging Face references

These aggregate-only controls use the cached BF16 Hugging Face snapshot
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52` and TT's lm-eval fork commit
`5416b8a97e8460cb74ae8cd96a457016fc0dc2e8` (`lm-eval 0.4.10.dev0`). Both
use the model chat template, seed 42, deterministic generation, the first 5%
of each task, and `max_length=262144`. No prompts or generated responses are
copied into the handoff.

## IFEval

- Task/version: `ifeval` v4, documents 0-27 (28/541).
- Completion contract: task default `max_gen_toks=1280`.
- CPU execution batch size: 4; wall time: 32m25s.
- Four metrics: prompt strict 85.7143%, instruction strict 88.3721%, prompt
  loose 85.7143%, instruction loose 88.3721%.
- TTI scalar (`score_task_keys_mean`): **87.0432%**.
- TT scalar: 82.62%. Subset-aware acceptance compares rounded effective
  correct counts: observed 23, threshold
  `floor(28 * 0.870432 * 0.95) = 23`; therefore PASS.

## GPQA CoT

- Task/version: `gpqa_diamond_cot_zeroshot` v1, documents 0-9 (10/198).
- Completion contract: `max_gen_toks=32768`; no output cap reduction.
- CPU execution batch size: 1; wall time: 49m31s.
- Flexible-extract exact match: **100.0%** (10/10).
- TT scalar: 40.0% (4/10). Subset-aware threshold is
  `floor(10 * 1.0 * 0.95) = 9`; therefore the current TT result FAILS and is a
  model-path correctness blocker, not a waiver candidate.

## Command shape

Both controls used:

```text
lm_eval --model hf \
  --model_args pretrained=<exact-snapshot>,dtype=bfloat16,trust_remote_code=True,max_length=262144 \
  --seed 42 --num_fewshot 0 --apply_chat_template \
  --trust_remote_code --confirm_run_unsafe_code --limit 0.05
```

IFEval used `--tasks ifeval --batch_size 4`. GPQA used
`--tasks gpqa_diamond_cot_zeroshot --batch_size 1 --gen_kwargs
max_gen_toks=32768`.
