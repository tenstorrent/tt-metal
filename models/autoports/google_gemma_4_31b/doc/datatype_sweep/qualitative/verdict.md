# Stage 08 qualitative verdict

Pass for datatype-sweep readiness.

The normal default path loaded `lm_head_bfp8_hifi2` from
`selected_precision_config.json`; its runtime summary reports an actual BFP8
LM-head tensor with HiFi2 and the selected physical attention/MLP dtypes. All
six prompts generated 64 TT tokens and the process closed all four devices
normally.

The scoped degeneracy checker exits zero. It finds no adjacent-duplication
failure and no repeated-trigram loop above its threshold. Manual review agrees:

- prompts 1 and 3 match HF token-for-token;
- prompts 0, 2, 4, and 5 differ from HF but remain grammatical, in-language,
  and semantically related base-model corpus continuations;
- prompt 1 repeats the supervised/unsupervised question, but HF repeats the
  exact same sequence, so it is not TT-only degeneration;
- prompt 2 repeats instruction-like corpus phrases, but the continuation is
  coherent and exactly matches the passing Stage 07 TT control;
- prompt 4 stays within French-translation training prompts and is more
  topically aligned than the Stage 07 question-list trajectory;
- prompt 5 gives a correct Fibonacci definition and sequence prefix, exactly
  matching the passing Stage 07 TT control.

Four of six TT outputs (prompts 1, 2, 3, and 5) are token-identical to the
Stage 07 selected-model controls. Prompts 0 and 4 take different but coherent
trajectories after lowering the LM head to BFP8; neither is degenerate.

The exact `GemmaTokenizer` has `chat_template=None`, so these are completion
controls rendered with `tokenizer.encode(..., add_special_tokens=True)`. Their
instruction-like autocomplete behavior is a checkpoint/prompt-format
limitation, not evidence of a TT runtime fallback or wrong-language failure.
See `degenerate_output_check.json`, `qualitative_prompt_format.json`, and
`vllm_qualitative_outputs.json`.
