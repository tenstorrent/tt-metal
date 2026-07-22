# Glossary

Supporting glossary for [`BENCHMARK.md`](BENCHMARK.md), for readers new to multimodal benchmarking or the Tenstorrent stack.

## Benchmarking terms

| Term | Meaning |
|------|---------|
| **Upstream** | The reference implementations/artifacts: the [Janus repo](https://github.com/deepseek-ai/Janus), the [Janus-Pro paper](https://arxiv.org/abs/2501.17811), the released [`deepseek-community/Janus-Pro-7B`](https://huggingface.co/deepseek-community/Janus-Pro-7B) weights, [MMBench](https://github.com/open-compass/MMBench), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), and [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval). |
| **Dataset / split** | A collection of samples (image + question + choices + expected answer), divided into predefined subsets (`train` / `val` / `dev` / `test`). |
| **Benchmark** | A dataset plus the evaluation procedure that measures model quality. |
| **Prompt template** | The fixed text scaffold that turns a raw sample into the exact string fed to the model (image placeholder, `A./B./C./D.` layout, answer instruction). Wording differs between frameworks, so it is a reproducibility variable. |
| **Evaluator** | Software that loads samples, builds prompts, calls the model, parses predictions, and scores them (e.g. VLMEvalKit, `lmms-eval`). |
| **CircularEval** | An MMBench option that rotates the answer options and re-asks the question to remove answer-position bias. CircularEval results are **not** comparable to single-ordering results. |
| **Image preprocessing** | Transforms applied before the vision encoder (RGB convert, resize, pad, normalize, to-tensor). Different implementations can change scores. |

## Constraint / precision terms

| Term | Meaning |
|------|---------|
| **Greedy decoding** | Pick the highest-probability token each step. Deterministic. (HF `do_sample=False`.) |
| **Precision / dtype** | The numeric format tensors use — `fp32` (32-bit), `bf16` (16-bit), or TT's `bfloat8_b` (`bf8`, a block-float format with a shared per-tile exponent). Lower precision is faster, less exact. |
| **Kernel** | A concrete hardware implementation of one operation. The same math can have several kernels; which runs, and in what order it sums, affects last-digit numerics. |
| **Math fidelity** | A Tenstorrent knob (LoFi → HiFi4) trading matmul accuracy for speed. No GPU equivalent. |
| **PCC** | Pearson Correlation Coefficient — the tt-metal metric for how close a device tensor is to a reference tensor. `1.0` = identical. |
| **Program config** | The fixed model definition on TT (op→kernel mapping, math fidelity, memory layout). It pins execution — there is no run-time kernel autotuning. |
| **Sampling nondeterminism** | Run-to-run variation from random token selection (`do_sample=True`). Removed by greedy. |
| **Numeric / backend nondeterminism** | Small differences from float rounding, op ordering, or kernel choice — present even under greedy, and differing across backends. |
