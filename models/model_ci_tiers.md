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
| vLLM Model Tests | Yes | Yes | Select&nbsp;models* |
| PR/Merge Gate Tests | Yes | — | — |
| Stress Tests | Yes | — | — |

\* vLLM Model Tests run as a **single pipeline** covering a curated set of models across all tiers at one cadence (see [vLLM Model Tests](#vllm-model-tests)). The tier shown per model there mirrors its End-to-End/Unit tier and does **not** gate execution — so a few Tier 3 models (e.g. Gemma-4-E2B) are included too.

## Tier 1 — Full Coverage
Tier 1 models receive the highest priority for developer time and bug fixing. They run the full CI test suite on a frequent schedule, including comprehensive sweeps, per-OP device performance tracking, commit-level gate tests, and multi-hour stress runs.

## Tier 2 — Standard Coverage
Tier 2 models share most test types with Tier 1 but at reduced scope. Sweep tests cover fewer sequence lengths and parameters. Stress tests, PR/merge gate tests, and device OP performance tests are excluded.

## Tier 3 — Minimum Coverage
Tier 3 models are compatible with the latest TT-Metal releases but are not optimized for best performance. Testing is kept to the minimum needed to confirm the models remain functional and accurate: an end-to-end accuracy check and a unit test for the most critical module.


# Current Model Assignments

The initial release of the 3-tier model CI includes models owned by the models-team. We plan to onboard the remaining models incrementally to reduce CI load. The current list of models and systems in the new pipelines can be seen below.


The Systems column is the union of systems across the End-to-End, Unit, and Sweep
configs for that model at that tier. A model may appear in more than one tier when
it is classified differently on different systems.

## Tier 1 Models
| Model | Systems |
|-------|---------|
| Llama3.1-8B | WH N150, BH P150 |
| Llama3.1-8B-DP | WH Galaxy |
| Llama3.3-70B | WH Galaxy |
| Qwen3-32B | WH Galaxy, BH Galaxy |
| Qwen3.6-27B | BH QuietBox 2 |
| GPT-OSS 120B | WH LLMBox, WH Galaxy, BH QuietBox 2, BH Galaxy |
| Gemma-4-12B | BH QuietBox 2 |
| Gemma-4-26B-A4B | BH QuietBox 2 |
| Gemma-4-31B | BH QuietBox 2 |
| Whisper | WH N150, BH P150 |
| Flux.1-schnell | BH QuietBox 2 |
| Flux.1-dev | BH QuietBox 2, BH Single Galaxy |
| Wan2.2-T2V-A14B | WH Galaxy |
| Wan2.2-I2V-A14B | WH Galaxy |
| TT-DiT (shared) | WH N150, BH QuietBox 2 |
## Tier 2 Models
| Model | Systems |
|-------|---------|
| GPT-OSS 20B | WH LLMBox, BH P150, BH QuietBox 2 |
| Llama3.1-8B | WH LLMBox, BH QuietBox 2 |
| Llama3.1-8B-DP | WH LLMBox, BH P300, BH QuietBox 2 |
| Llama3.3-70B | WH LLMBox, BH QuietBox 2 |
| Llama3.2-90B-Vision | WH LLMBox |
| Qwen3-32B | WH LLMBox, BH QuietBox 2 |
| Qwen2.5-32B | WH LLMBox, BH QuietBox 2 |
| Qwen2.5-Coder-32B | WH LLMBox |
| Qwen2.5-VL-72B | WH LLMBox, BH QuietBox 2 |
| Qwen3-VL-32B | WH LLMBox |
| Shallow-UNet | WH N150, WH LLMBox, BH P150 |
| Mistral-7B | WH N150 |
| Mixtral-8x7B | WH LLMBox |
| Gemma-3-4B | WH N150, BH P150 |
| Gemma-3-27B | WH LLMBox |
| Gemma-4-26B-A4B | WH LLMBox |
| Gemma-4-31B | WH LLMBox |
| Stable Diffusion XL | WH N150, WH N300, BH P150 |
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
| Qwen2.5-72B | WH LLMBox |
| Qwen2.5-VL-32B | WH LLMBox, BH QuietBox 2 |
| Gemma-4-E2B | WH N150, BH P150 |
| Gemma-4-E4B | BH P300, BH QuietBox 2 |
| Mamba-2.8B | WH N150 |
| Phi-3-mini | WH N150 |
| Janus-Pro-7B | BH P150 |
| Panoptic-DeepLab | BH P150 |
| BEVFormer | BH P150 |


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

## vLLM Model Tests

End-to-end serving tests that stand up a vLLM server for each model and exercise the
tt-metal backend through the OpenAI-compatible API. Unlike the pipelines above, this
is **not** split by tier — every entry runs together as a single pipeline at one
nightly cadence, since the curated model set is small enough that per-tier scheduling
would be overkill. Each entry still carries a `tier` value for classification, mirroring
the model's End-to-End/Unit tier.

Per entry the pipeline runs a throughput/latency benchmark and, where enabled, a
coherence guard (verbatim echo), structured-output benchmark, multimodal image test,
and the sampling-parameter test suite. A fast `validate-filters` pre-check fails in
seconds (before the build) if a manually-selected model/SKU/tier combination matches
no test, printing the valid combinations.

| Item | Workflow |
|------|----------|
| Pipeline | [`vLLM Model Tests`](../.github/workflows/vllm-model-tests.yaml) |
| Impl | [`[internal] vLLM Model tests impl`](../.github/workflows/vllm-model-tests-impl.yaml) |
| Config | [`vllm_model_tests.yaml`](../tests/pipeline_reorg/vllm_model_tests.yaml) |

Models covered (system · classification tier):

| Model | Systems | Tier |
|-------|---------|------|
| Llama3.1-8B | WH Galaxy | 1 |
| Llama3.1-8B | WH LLMBox, BH QuietBox 2, BH LoudBox | 2 |
| Llama3.3-70B | WH Galaxy | 1 |
| Qwen3-32B | WH Galaxy | 1 |
| Qwen3.6-27B | BH QuietBox 2 | 1 |
| Qwen2.5-VL-72B | WH LLMBox | 2 |
| Qwen3-VL-32B | WH LLMBox | 2 |
| GPT-OSS 120B | WH Galaxy | 1 |
| Gemma-3-27B | WH LLMBox, WH Galaxy | 2 |
| Gemma-4-12B | BH QuietBox 2 | 1 |
| Gemma-4-26B-A4B | BH QuietBox 2 | 1 |
| Gemma-4-31B | BH QuietBox 2 | 1 |
| Gemma-4-31B | WH LLMBox | 2 |
| Gemma-4-E2B | WH N150 | 3 |
| NoOp (vLLM overhead) | WH Galaxy | 1 |

## Other Pipelines

> **TODO:** Expand descriptions for the pipelines below.

| Pipeline | Workflow |
|----------|----------|
| PR Gate | [`PR Gate`](../.github/workflows/pr-gate.yaml) |
| Merge Gate | [`Merge Gate`](../.github/workflows/merge-gate.yaml) |
| Galaxy stress tests | [`(Galaxy) Stress`](../.github/workflows/galaxy-stress-tests.yaml) |
