# Llama 3.1 8B Instruct Autoport

Status: full-model stage complete as of 2026-06-15. vLLM integration was not
started.

Full-model batch-1 performance on T3K `1x8` Ring:

| Path | Prompt | TTFT | Decode |
| --- | --- | ---: | ---: |
| Teacher-forcing readiness | AIME24 chat-template, 184 prompt tokens, 100 generated tokens | 1094.60 ms | 22.18 t/s/u decode, 17.99 t/s/u e2e |
| Token-out traced generation | story prompt, 60 prompt tokens, 100 generated tokens | 616.64 ms | 49.65 t/s/u including first trace capture; 69.21 t/s/u steady replay |

Accuracy gates:

| Gate | Result |
| --- | --- |
| `run_prefill_check` | top1 90/100, top5 100/100, top100 100/100 |
| `run_teacher_forcing` | top1 92/100, top5 100/100, top100 100/100 |
| `run_autoregressive` | HF and TT each generated 100 tokens; TT output is coherent English with no repetition or wrong-language drift |

The full-model implementation is in `tt/model.py` and `tt/generator.py`.
Detailed commands, trace evidence, profiler artifacts, and fallback audit are in
`doc/full_model/README.md` and `doc/full_model/work_log.md`.
