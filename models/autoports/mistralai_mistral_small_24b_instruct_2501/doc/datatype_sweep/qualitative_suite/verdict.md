# Selected-precision qualitative verdict

Verdict: pass.

The exact checkpoint tokenizer has a non-empty chat template, so all six shared
prompts were rendered with `apply_chat_template(..., add_generation_prompt=True)`.
HF and TT used the same rendered token IDs, greedy decoding, and maximum 128
new tokens. The selected TT policy produced output lengths
`[18, 128, 128, 128, 62, 128]` versus HF `[16, 128, 128, 128, 62, 128]`.

Every saved completion was inspected. Prompt 1 is a valid topical haiku;
prompts 2 and 4 are coherent explanations; prompt 3 is a coherent story;
prompt 5 gives the same correct French translation as HF; and prompt 6 emits
coherent Fibonacci Python. There is no wrong language, prompt echo, mechanical
repetition, doubled text, control-token leakage, or gibberish. Prompt 1 repeats
exactly under greedy decoding. Differences from HF are ordinary valid greedy
continuation differences and not visible quality regressions.

Evidence: `qualitative_prompt_format.json`, `suite_summary.json`, and each
`prompt_*/{rendered_prompt,hf_completion,tt_completion,autoregressive_meta}`
artifact.
