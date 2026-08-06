# Glossary

Supporting glossary for [`README.md`](README.md).

## Benchmarking terms

| Term | Meaning |
|------|---------|
| **Upstream** | Reference artifacts: [Janus](https://github.com/deepseek-ai/Janus), the [paper](https://arxiv.org/abs/2501.17811), [`deepseek-community/Janus-Pro-7B`](https://huggingface.co/deepseek-community/Janus-Pro-7B), [MMBench](https://github.com/open-compass/MMBench), [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval). |
| **Dataset / split** | Sample collection and which subset is used, e.g. MMBench `en` + `dev` → task `mmbench_en_dev`. |
| **Evaluator** | Loads samples, builds prompts, calls the model, scores predictions (`lmms-eval`, VLMEvalKit). |
| **Prompt template** | Fixed text scaffold around a sample (image placeholder, `A./B./C./D.` layout, “answer with the letter…” instruction, chat roles). Different templates → different scores on the same dataset. |
| **MCQ** | **Multiple-choice question.** The model sees the question plus labeled options (A/B/C/D/…) and is asked to answer with a letter. MMBench is MCQ; our community / `lmms-eval` path is **single-pass** MCQ (options shown once, no CircularEval). |
| **Post-prompt** | Trailing instruction appended by the evaluator (MMBench default: `Answer with the option's letter from the given choices directly.`). |
| **Chat / conversation template** | Model-specific wrapping of the evaluator prompt into roles (`` `<\|User\|>` `` / `` `<\|Assistant\|>` `` for Janus). Applied by the serving stack or a native model adapter — separate from the MMBench option format. |
| **CircularEval** | MMBench option that rotates answer choices and re-asks to reduce position bias. Scores **with** CircularEval are not comparable to single-pass scores. |
| **Image preprocessing** | Resize / pad / normalize before the vision encoder. Janus: 384×384, mean/std `[0.5,0.5,0.5]`, pad `[127,127,127]`. |
| **Scoring / answer extraction** | How raw model text becomes a letter: rule-based match (`can_infer_option` / `can_infer_text`), then optional GPT fallback. Must be the same path on both sides. |
| **ISL / OSL** | Input / output sequence length (tokens). Primary axes for **perf** benchmarks. |
| **TTFT** | Time to first token (vision + prefill for multimodal). |
| **tput_user** | Decode tokens per second per user (concurrency=1 ≈ single-stream decode speed). |
| **Theoretical target** | Estimated ceiling for a model×device in tt-inference-server `model_performance_reference.json`. |

## Decoding / tokens

| Term | Meaning |
|------|---------|
| **Greedy decoding** | Always pick the highest-probability next token (`do_sample=False`). Deterministic. Required for accuracy. |
| **EOS token** | End-of-sequence token that stops generation. Janus: `<｜end▁of▁sentence｜>` (id **100001**). Not Llama’s `` `<\|end_of_text\|>` `` / `` `<\|eot_id\|>` ``. |
| **Stop string(s)** | Extra strings that halt decode (API `stop=`). Janus conversation template also stops on `` `<\|User\|>` `` so the model does not open a new turn. |
| **BOS token** | Begin-of-sequence. Janus: `<｜begin▁of▁sentence｜>` (id **100000**). |
| **max_new_tokens** | Cap on how many **new** tokens the model may generate in one reply (prompt tokens not counted). Generation also stops earlier on EOS / stop strings. Example: `max_new_tokens=512`. For MMBench letter answers this almost never binds; mismatch vs 1024 is low impact unless the cap is set very low. |

## Constraint / precision terms

| Term | Meaning |
|------|---------|
| **Precision / dtype** | Numeric format — `fp32`, `bf16`, or TT `bfloat8_b` (`bf8`). |
| **Kernel** | Concrete hardware implementation of one op; ordering/rounding differs across backends. |
| **Math fidelity** | TT knob (LoFi → HiFi4) trading matmul accuracy for speed. |
| **PCC** | Pearson Correlation Coefficient — TT tensor vs reference tensor. `1.0` = identical. Fidelity gate, not task accuracy. |
| **Program config** | Fixed TT model definition (op→kernel, fidelity, layout). No per-run kernel autotuning. |
| **Sampling nondeterminism** | Run-to-run noise from `do_sample=True`. Removed by greedy. |
| **Numeric / backend nondeterminism** | Small float/ordering differences across backends even under greedy. |
