# microsoft/Phi-3.5-mini-instruct TTNN Autoport

Optimized full-model status, 2026-06-15:

| path | TTFT | trace-verified decode |
| --- | ---: | ---: |
| teacher-forcing decode, AIME24 chat reference | 226.93 ms | 40.15 t/s/u |
| token-out greedy decode, prompt128/gen128, no readback | 254.47 ms | 56.43 t/s/u |

Both measurements are batch-1 on the 1x8 T3K ring mesh with canonical split sampling. Teacher forcing and token-out are intentionally reported separately: teacher forcing copies the forced token into the persistent traced input between decode steps, while token-out uses sampled-token device feedback. The token-out benchmark has no per-token sampled-token readback or full-logit readback.

Readiness accuracy against `readiness_aime24_chat_100.refpt`:

| check | top-1 | top-5 | top-100 |
| --- | ---: | ---: | ---: |
| prefill | 96/100 | 100/100 | 100/100 |
| traced teacher forcing | 91/100 | 100/100 | 100/100 |

Qualitative autoregressive artifacts are under `readiness_autoregressive/`. The final TT completion is coherent and has no mechanical repetition, wrong-language drift, or empty/early-stop behavior. The current CPU HF greedy reference repeated `candid`, so HF/TT token agreement is informational only. `degenerate_report.json` reports no TT degenerate output.

Implementation entrypoints:

- `tt/model.py`: full CausalLM path around the optimized multichip decoder stack.
- `tt/generator.py`: standard readiness `build_generator` contract, traced decode, split sampling, KV-cache/page-table ownership, and reset handling.
- `doc/optimized_full_model/README.md`: optimized full-model commands, metrics, sampler contract, lower-bound accounting, fallback audit, watcher evidence, limitations, and artifacts.
- `doc/full_model/README.md`: commands, metrics, trace evidence, fallback audit, limitations, and artifacts.
