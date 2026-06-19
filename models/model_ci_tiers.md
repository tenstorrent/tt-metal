# Model CI Testing Tiers

This document aims to explain our new approach to internal testing of supported models. It does not aim to replace the list of supported models in Tenstorrent hardware. Please refer to the main README file for that.

TT-Metal uses a 3-tier system to classify models for CI testing. Each tier defines a different level of test coverage and frequency, allowing us to prioritize CI/CD resources where they matter most while maintaining broad coverage across the model portfolio.

All models deemed supported by TT-Metal are guaranteed to be compatible with the latest software release, regardless of their tier. If a bug is introduced that breaks model compatibility, it will be addressed in order of priority according to its tier.

The current classification addresses both models and target systems. As such, the same model might belong to different tiers, if targeting different systems. In many cases, a model is compatible across our full range of systems and architectures, even if it is not explicitly listed here.

# Definitions

## Coverage at a Glance

| Test Type | Tier 1 | Tier 2 | Tier 3 |
|-----------|--------|--------|--------|
| End-to-End Tests | Full (perf, determinism, accuracy) | Full | Accuracy only |
| Unit Tests | All required modules | All required modules | Most critical module only |
| Sweep Tests | Comprehensive | Reduced | — |
| Device OP Perf Tests | Yes | — | — |
| vLLM Integration Tests | Yes | Yes | — |
| PR/Merge Gate Tests | Yes | — | — |
| Stress Tests | Yes | — | — |

## Tier 1 — Full Coverage
Tier 1 models receive the highest priority for developer time and bug fixing. They run the full CI test suite on a frequent schedule, including comprehensive sweeps, per-OP device performance tracking, commit-level gate tests, and multi-hour stress runs.

## Tier 2 — Standard Coverage
Tier 2 models share most test types with Tier 1 but at reduced scope. Sweep tests cover fewer sequence lengths and parameters. Stress tests, PR/merge gate tests, and device OP performance tests are excluded.

## Tier 3 — Minimum Coverage
Tier 3 models are compatible with the latest TT-Metal releases but are not optimized for best performance. Testing is kept to the minimum needed to confirm the models remain functional and accurate: an end-to-end accuracy check and a unit test for the most critical module.


# Current Model Assignments

The initial release of the 3-tier model CI includes models owned by the models-team. We plan to onboard the remaining models incrementally to reduce CI load. The current list of models and systems in the new pipelines can be seen below.


## Tier 1 Models
| Model | Systems |
|-------|---------|
| Llama3.1-8B | WH N150, BH P150 |
| Llama3.1-8B-DP | WH Galaxy |
| Llama3.3-70B | WH Galaxy |
| Qwen3-32B | WH Galaxy |
| GPT-OSS 120B | WH Galaxy, BH Galaxy, BH QuietBox 2 |
| Whisper | WH N150, BH P150 |
| Gemma-4-12B | BH QuietBox 2 |
| Gemma-4-26B-A4B | BH QuietBox 2 |
| Gemma-4-31B | BH QuietBox 2 |
## Tier 2 Models
| Model | Systems |
|-------|---------|
| GPT-OSS 20B | WH LLMBox, BH P150, BH QuietBox 2 |
| Llama3.1-8B | WH LLMBox, BH QuietBox 2 |
| Llama3.1-8B-DP | WH LLMBox, BH P300 |
| Llama3.3-70B | WH LLMBox, BH QuietBox 2 |
| Qwen3-32B | WH LLMBox, BH QuietBox 2 |
| Qwen2.5-32B | WH LLMBox, BH QuietBox 2 |
| Qwen2.5-Coder-32B | WH LLMBox |
| Qwen2.5-72B-VL | WH LLMBox, BH QuietBox 2 |
| Llama90B-VL | WH LLMBox |
| Shallow-UNet | WH N150, WH LLMBox |
| Mistral-7B | WH N150 |
| Mixtral-8x7B | WH LLMBox |
| Gemma-3-4B | WH N150 |
| Gemma-3-27B | WH LLMBox |
| Gemma-4-E2B | WH N150, BH P150 |
| Gemma-4-E4B | WH N150, BH P150, BH P300, BH QuietBox 2 |
| Gemma-4-26B-A4B | WH LLMBox |
| Gemma-4-31B | WH LLMBox |
## Tier 3 Models
| Model | Systems |
|-------|---------|
| Falcon-7B | WH N150 |
| Falcon-40B | WH LLMBox |
| Llama3.2-1B | WH N150 |
| Llama3.2-3B | WH N150 |
| Llama3.2-11B-Vision | WH LLMBox |
| QwQ-32B | WH LLMBox |
| Qwen2.5-7B | WH N300 |
| Qwen2.5-VL-32B | WH LLMBox, BH QuietBox 2 |
| Mamba-2.8B | WH N150 |
| Phi-3-mini | WH N150 |


# Pipelines

Each test type has a per-tier GitHub Actions workflow and a shared configuration file. The workflows are separated by tier to allow independent scheduling, but the test definitions live in a single config YAML per pipeline.

## End-to-End Tests

This pipeline covers two areas:
- **Performance evaluation** — Runs multiple batch-32 requests, reports TTFT and tokens/sec/user, and validates determinism by comparing outputs across batches.
- **Accuracy validation** — Uses teacher forcing against a reference to report top-1/top-5 accuracy against pre-established targets.

| Tier | Workflow |
|------|----------|
| Tier 1 | [`(Tier 1) Models End-To-End Tests`](../.github/workflows/models-t1-e2e-tests.yaml) |
| Tier 2 | [`(Tier 2) Models End-To-End Tests`](../.github/workflows/models-t2-e2e-tests.yaml) |
| Tier 3 | [`(Tier 3) Models End-To-End Tests`](../.github/workflows/models-t3-e2e-tests.yaml) |
| Config | [`models_e2e_tests.yaml`](../tests/pipeline_reorg/models_e2e_tests.yaml) |

## Unit Tests

Short module-level tests that compare PCC against the reference implementation. Tier 1 and 2 models should cover most components; Tier 3 requires only the most critical module (typically a single decoder layer for LLMs).

| Tier | Workflow |
|------|----------|
| Tier 1 | [`(Tier 1) Models Unit Tests`](../.github/workflows/models-t1-unit-tests.yaml) |
| Tier 2 | [`(Tier 2) Models Unit Tests`](../.github/workflows/models-t2-unit-tests.yaml) |
| Tier 3 | [`(Tier 3) Models Unit Tests`](../.github/workflows/models-t3-unit-tests.yaml) |
| Config | [`models_unit_tests.yaml`](../tests/pipeline_reorg/models_unit_tests.yaml) |

## Sweep Tests

Parameter sweeps across the configurations each model supports. Currently includes:
- **Sequence length** — From short contexts up to the model's maximum supported length
- **Sampling parameters** — Logprobs, seeds, penalties, and other sampling features
- **Prefetcher** — Runs with prefetcher enabled and disabled to isolate model-specific issues

| Tier | Workflow |
|------|----------|
| Tier 1 | [`(Tier 1) Models Sweep Tests`](../.github/workflows/models-t1-sweep-tests.yaml) |
| Tier 2 | [`(Tier 2) Models Sweep Tests`](../.github/workflows/models-t2-sweep-tests.yaml) |
| Config | [`models_sweep_tests.yaml`](../tests/pipeline_reorg/models_sweep_tests.yaml) |

> **Note:** Tier 3 does not include sweep tests.

## Device Perf Tests

Captures device timing for a single layer of each target model, used to track performance for models under active optimization. Tier 1 only.

> **TODO:** Add pipeline links once merged to main.

## Other Pipelines

> **TODO:** Expand descriptions for the pipelines below.

| Pipeline | Workflow |
|----------|----------|
| PR Gate | [`PR Gate`](../.github/workflows/pr-gate.yaml) |
| Merge Gate | [`Merge Gate`](../.github/workflows/merge-gate.yaml) |
| vLLM nightly tests | [`vLLM nightly tests`](../.github/workflows/vllm-nightly-tests.yaml) |
| vLLM nightly impl | [`[internal] vLLM nightly tests impl`](../.github/workflows/vllm-nightly-tests-impl.yaml) |
| Galaxy stress tests | [`(Galaxy) Stress`](../.github/workflows/galaxy-stress-tests.yaml) |
| TTNN stress tests | [`ttnn stress tests`](../.github/workflows/ttnn-stress-tests.yaml) |
