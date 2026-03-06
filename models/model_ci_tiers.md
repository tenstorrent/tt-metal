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
| Llama3.3-70B | WH Galaxy, BH Galaxy |
| Qwen3-32B | WH Galaxy, BH Galaxy |
| Deepseek | WH Galaxy |
| GPT-OSS | WH Galaxy |
| llama3.1-8B | N150, P150 |
| Whisper | N150, P150 |
## Tier 2 Models
| Model | Systems |
|-------|---------|
| GPT-OSS 120B batch=1 | WH Galaxy |
| GPT-OSS 20B | WH LoudBox |
| Llama70B (TTT) | WH/BH LoudBox |
| Qwen3-32B (TTT) | WH/BH LoudBox |
| Qwen2.5 32B (TTT) | WH LoudBox |
| llama90B-VL | WH LoudBox |
| Qwen2.5-72B-VL | WH LoudBox |
| Shallow-UNet | N150 |
| Mistral-7B | N150 |
| Mixtral 8x7B | WH LoudBox |
| Gemma-3-27B | WH LoudBox |
## Tier 3 Models
| Model | Systems |
|-------|---------|
| Falcon-7B | N150 |
| Falcon-40B | WH LoudBox |
| Mamba | N150 |
| Llama 1B | N150 |
| Llama 3B | N150 |
| Llama 11B text | N150 |
| Llama 11B-VL | WH LoudBox |
| Qwen 2.5B-7B | N150 |
| Qwen 2.5B-72B | WH LoudBox |
| Qwen2.5-VL-32B-Instruct | WH LoudBox |
| Phi 1.5 | N150 |
| Phi 1.4 | N150 |


# Pipelines
The new models CI pipelines are divided into different tiers, to facilitate automated schedule on github actions. Each test type has a per-tier dispatch workflow and a shared internal implementation workflow.

Although pipelines are separated by tiers to facilitate github actions integration, we provide a single configuration yaml file per group of pipelines. This can be found in the sub-sections below.

## Models End-to-End Tests Pipeline
This pipeline mainly includes two types of tests:
- **End-to-end performance evaluation**
  - Typically runs multiple batch-32 requests
  - Reports end-to-end performance (time-to-first-token and tokens/sec/user) after first batch
  - Validates determism by comparing the different users outputs across different batches
- **End-to-end accuracy validation**
  - Teacher forcing against reference, reports final top-1/top-5 accuracy
  - Validates against pre-estabilished targets


| Tier | Workflow |
|------|----------|
| Tier 1 | [`[Tier 1] Models End-To-End Tests`](../.github/workflows/models-t1-e2e-tests.yaml) |
| Tier 2 | [`[Tier 2] Models End-To-End Tests`](../.github/workflows/models-t2-e2e-tests.yaml) |
| Tier 3 | [`[Tier 3] Models End-To-End Tests`](../.github/workflows/models-t3-e2e-tests.yaml) |
| Config | [`models_e2e_tests.yaml`](../tests/pipeline_reorg/models_e2e_tests.yaml) |

## Models Unit Tests Pipeline

| Tier | Workflow |
|------|----------|
| Tier 1 | [`[Tier 1] Models Unit Tests`](../.github/workflows/models-t1-unit-tests.yaml) |
| Tier 2 | [`[Tier 2] Models Unit Tests`](../.github/workflows/models-t2-unit-tests.yaml) |
| Tier 3 | [`[Tier 3] Models Unit Tests`](../.github/workflows/models-t3-unit-tests.yaml) |
| Config | [`models_unit_tests.yaml`](../tests/pipeline_reorg/models_unit_tests.yaml) |

## Models Sweep Tests Pipeline

| Tier | Workflow |
|------|----------|
| Tier 1 | [`[Tier 1] Models Sweep Tests`](../.github/workflows/models-t1-sweep-tests.yaml) |
| Tier 2 | [`[Tier 2] Models Sweep Tests`](../.github/workflows/models-t2-sweep-tests.yaml) |
| Config | [`models_sweep_tests.yaml`](../tests/pipeline_reorg/models_sweep_tests.yaml) |

> **Note:** Tier 3 does not include sweep tests.

## Models Device Perf Tests Pipeline

| Pipeline | Workflow |
|----------|----------|
| Device perf regressions | [`(Single-card) Device perf regressions`](../.github/workflows/perf-device-models.yaml) |
| Device perf impl | [`[internal] Single-card Device perf regressions impl`](../.github/workflows/perf-device-models-impl.yaml) |
| Model perf tests | [`(Single-card) Model perf tests`](../.github/workflows/perf-models.yaml) |
| Model perf impl | [`[internal] Perf models impl`](../.github/workflows/perf-models-impl.yaml) |

> **Note:** Device perf tests are Tier 1 only.

## Other Pipelines

| Pipeline | Workflow |
|----------|----------|
| Models post-commit | [`[internal] models tests impl`](../.github/workflows/models-post-commit.yaml) |
| PR Gate | [`PR Gate`](../.github/workflows/pr-gate.yaml) |
| Merge Gate | [`Merge Gate`](../.github/workflows/merge-gate.yaml) |
| vLLM nightly tests | [`vLLM nightly tests`](../.github/workflows/vllm-nightly-tests.yaml) |
| vLLM nightly impl | [`[internal] vLLM nightly tests impl`](../.github/workflows/vllm-nightly-tests-impl.yaml) |
| Galaxy stress tests | [`(Galaxy) Stress`](../.github/workflows/galaxy-stress-tests.yaml) |
| TTNN stress tests | [`ttnn stress tests`](../.github/workflows/ttnn-stress-tests.yaml) |
