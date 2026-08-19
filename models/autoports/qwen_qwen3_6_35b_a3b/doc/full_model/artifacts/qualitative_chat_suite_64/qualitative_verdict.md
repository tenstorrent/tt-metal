# Qualitative Chat Suite

- Model: `Qwen/Qwen3.6-35B-A3B`
- Prompt format: HF tokenizer chat template, `add_generation_prompt=True`
- Prompts: 6
- Max new tokens: 64

| Prompt | Prompt tokens | HF tokens | TT tokens | TT trace |
| ---: | ---: | ---: | ---: | --- |
| 0 | 18 | 64 | 64 | True (63 replays) |
| 1 | 25 | 64 | 64 | True (63 replays) |
| 2 | 33 | 64 | 64 | True (63 replays) |
| 3 | 19 | 64 | 64 | True (63 replays) |
| 4 | 24 | 64 | 64 | True (63 replays) |
| 5 | 20 | 64 | 64 | True (63 replays) |

Manual verdict: pass.

Prompt 0 through prompt 4 match the HF greedy continuation exactly for all 64
generated tokens. Prompt 5 diverges from HF at generated token 44, but the TT
completion remains coherent on the requested Python/Fibonacci task and does not
show repetition, wrong-language drift, or early divergence symptoms.

`logs/check_degenerate_output_qualitative_chat_suite_64.log` reports no
degenerate output across the six traced completions.
