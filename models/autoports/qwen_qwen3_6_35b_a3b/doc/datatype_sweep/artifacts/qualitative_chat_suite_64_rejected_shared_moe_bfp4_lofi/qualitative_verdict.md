# Rejected Qualitative Chat Suite

- Model: `Qwen/Qwen3.6-35B-A3B`
- Prompt format: HF tokenizer chat template, `add_generation_prompt=True`
- Candidate config: `shared_moe_bfp4_lofi`
- Verdict: rejected
- Reason: patched degenerate-output checker found long repeated punctuation and raw token-id runs on prompts 1 and 4.

This directory preserves the qualitative evidence that caused `shared_moe_bfp4_lofi` to be excluded from final selection even though it was the fastest traced teacher-forcing accuracy-pass candidate.

Critical findings:

- Prompt 1: repeated `!` run length 24 and raw token-id run length 24.
- Prompt 4: repeated `!` run length 23 and raw token-id run length 23.

Artifacts:

- `vllm_qualitative_outputs.json`
- `tt_qualitative_outputs.json`
- `rejected_shared_moe_bfp4_lofi_repetition_report.json`
- `hf_qualitative_outputs.json`
- `qualitative_prompt_format.json`
- `qualitative_prompts.json`
