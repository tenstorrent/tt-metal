# Qualitative Chat Suite

- Model: `Qwen/Qwen3.6-35B-A3B`
- Prompt format: HF tokenizer chat template, `add_generation_prompt=True`
- HF control: reused from `doc/full_model/artifacts/qualitative_chat_suite_64/hf_qualitative_outputs.json`
- TT config: `baseline_default`, loaded from `doc/datatype_sweep/selected_precision_config.json`
- Prompts: 6
- Max new tokens: 64

| Prompt | Prompt tokens | HF tokens | TT tokens | TT trace | Degenerate check |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 18 | 64 | 64 | True (63 replays) | clean |
| 1 | 25 | 64 | 64 | True (63 replays) | clean |
| 2 | 33 | 64 | 64 | True (63 replays) | clean, adjacent duplication 0.0588 |
| 3 | 19 | 64 | 64 | True (63 replays) | clean |
| 4 | 24 | 64 | 64 | True (63 replays) | clean |
| 5 | 20 | 64 | 64 | True (63 replays) | clean |

Manual verdict: pass. The TT outputs are chat-template continuations with no wrong-language output, control-token leakage, cross-request leakage, gibberish, repeated punctuation run, or mechanical repeated-token failure. Prompt 2 has a small word-level adjacent-duplication rate below the 0.10 critical threshold; raw token-id max run is 2.

The rejected `shared_moe_bfp4_lofi` candidate is preserved separately in `../qualitative_chat_suite_64_rejected_shared_moe_bfp4_lofi/`; its patched checker report records critical repeated punctuation and token-id runs on prompts 1 and 4.

Artifacts:

- `qualitative_prompt_format.json`
- `qualitative_prompts.json`
- `hf_qualitative_outputs.json`
- `tt_qualitative_outputs.json`
- `vllm_qualitative_outputs.json`
- `degenerate_output_report.json`
