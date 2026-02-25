# Needed CI Tests for PR #38139
# Warmup calls in metal tests (same as vLLM)

---

## models/common/demos/llama31_8B_demo.py

**CI Workflow:** `t3000-e2e-tests-impl.yaml`
**Pipeline config:** `tests/pipeline_reorg/t3k_e2e_tests.yaml`
**Job name:** `models_tttv2_llama31_8B_tests`

```
MESH_DEVICE=T3K HF_HUB_OFFLINE=1 HF_HOME=/mnt/MLPerf/huggingface \
HF_MODEL=meta-llama/Llama-3.1-8B-Instruct \
TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama--Llama-3.1-8B-Instruct \
pytest --tb=short models/common/demos/llama31_8B_demo.py
```

---

## models/demos/gpt_oss/demo/text_demo.py
## models/demos/gpt_oss/tests/accuracy/test_model.py

**CI Workflow:** `t3000-demo-tests-impl.yaml` (T3K) + `galaxy-unit-tests-impl.yaml` (Galaxy)
**Pipeline configs:** `tests/pipeline_reorg/t3k_demo_tests.yaml`, `tests/pipeline_reorg/galaxy_demo_tests.yaml`
**Shell script:** `tests/scripts/t3000/run_t3000_demo_tests.sh` → `run_t3000_gpt_oss_tests`
**Job names:** `t3k_gpt_oss_tests` (T3K), Galaxy GPT-OSS job (Galaxy)

T3K (1x8):
```
pytest models/demos/gpt_oss/demo/text_demo.py -k "1x8" --timeout 1000
pytest models/demos/gpt_oss/tests/accuracy/test_model.py -k "1x8" --timeout 900
```
Galaxy (4x8):
```
pytest models/demos/gpt_oss/demo/text_demo.py -k "4x8" --timeout 1000
pytest models/demos/gpt_oss/tests/accuracy/test_model.py -k "4x8" --timeout 900
```

---

## models/demos/llama3_70b_galaxy/demo/text_demo.py
## models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py

**CI Workflow:** `galaxy-unit-tests-impl.yaml`
**Pipeline config:** `tests/pipeline_reorg/galaxy_demo_tests.yaml`

```
pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "repeat" --timeout 1000
pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "pcc-80L" --timeout 1000
pytest models/demos/llama3_70b_galaxy/demo/text_demo.py -k "batch-32-non-uniform-sampling" --timeout 500
pytest models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py -k "batch-32" --timeout 1000
pytest models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py -k "repeat2" --timeout 1000
pytest models/demos/llama3_70b_galaxy/demo/text_qwen_demo.py -k "long-8k-b1" --timeout 1000
```

---

## models/demos/multimodal/gemma3/demo/text_demo.py
## models/demos/multimodal/gemma3/demo/vision_demo.py

**CI Workflow:** `t3000-demo-tests-impl.yaml` (T3K) + `single-card-demo-tests-impl.yaml` (N300)
**Pipeline config:** `tests/pipeline_reorg/t3k_demo_tests.yaml`
**Shell script:** `tests/scripts/t3000/run_t3000_demo_tests.sh` → `run_t3000_gemma3_tests`
**Job names:** `t3k_gemma3_tests` (T3K), `gemma3` N300 (single-card)

```
pytest models/demos/multimodal/gemma3/demo/text_demo.py -k "performance and ci-1"
pytest models/demos/multimodal/gemma3/demo/vision_demo.py -k "performance and batch1-multi-image-trace"
```

---

## models/demos/qwen25_vl/demo/demo.py

**CI Workflow:** `t3000-demo-tests-impl.yaml` (T3K) + `single-card-demo-tests-impl.yaml` (N150/N300)
**Pipeline config:** `tests/pipeline_reorg/t3k_demo_tests.yaml`
**Shell script:** `tests/scripts/t3000/run_t3000_demo_tests.sh` → `run_t3000_qwen25_vl_tests`
**Job names:** `t3k_qwen25_vl_tests` (T3K), `qwen25_vl` N150/N300 (single-card)

```
pytest models/demos/qwen25_vl/demo/demo.py --timeout 900
```

---

## models/demos/qwen3_vl/demo/demo.py

**CI Workflow:** `t3000-demo-tests-impl.yaml`
**Pipeline config:** `tests/pipeline_reorg/t3k_demo_tests.yaml`
**Shell script:** `tests/scripts/t3000/run_t3000_demo_tests.sh` → `run_t3000_qwen3_vl_tests`
**Job name:** `t3k_qwen3_vl_tests`

```
pytest models/demos/qwen3_vl/demo/demo.py --timeout 600
```

---

## models/demos/tg/llama3_70b/demo/demo.py

**CI Workflow:** `galaxy-unit-tests-impl.yaml`
**Pipeline config:** `tests/pipeline_reorg/galaxy_demo_tests.yaml`
**Note:** Only has `# ovde se init generator` placeholder comment added — old Llama generator,
not WarmupForwardMixin. Real warmup handled internally by `demo_warmup()`. Low priority.

---

## models/tt_transformers/demo/simple_text_demo.py

**CI Workflows:** `t3000-demo-tests-impl.yaml` (T3K) + `single-card-demo-tests-impl.yaml` (N150/N300)
             + `galaxy-unit-tests-impl.yaml` (Galaxy)
**Shell scripts:** `run_t3000_demo_tests.sh` + `run_single_card_demo_tests.sh`
**Functions:** `run_t3000_llama3_tests`, `run_t3000_llama3_70b_tests`, `run_t3000_qwen25_tests`,
              `run_t3000_mistral_tests`, `run_t3000_mixtral_tests`, `run_llama3_func`, `run_llama3_perf`

This file is exercised by virtually every model's demo test — running any of the
t3k_demo_tests or single-card demo tests will cover it.

---

## models/tt_transformers/demo/simple_vision_demo.py

**CI Workflow:** `t3000-demo-tests-impl.yaml`
**Shell script:** `tests/scripts/t3000/run_t3000_demo_tests.sh` → `run_t3000_llama3_vision_tests`
**Job names:** `t3k_llama3_vision_tests`, `t3k_llama3_90b_vision_tests`

```
pytest models/tt_transformers/demo/simple_vision_demo.py -k "not batch1-notrace"
pytest models/tt_transformers/demo/simple_vision_demo.py -k "batch1-trace"
```

---

## models/tt_transformers/demo/multimodal_demo_chat.py
## models/tt_transformers/demo/multimodal_demo_text.py

**CI Workflow:** NOT IN ANY CI PIPELINE
**Note:** Only has `# ovde se init generator` placeholder — vision warmup not yet implemented.
No CI pipeline references found. No action needed until vision warmup is implemented.

---

## models/tt_transformers/tests/test_model_prefill.py
## models/tt_transformers/tests/test_chunked_generation.py

**CI Workflow:** `t3000-integration-tests-impl.yaml`
**Shell script:** `tests/scripts/t3000/run_t3000_integration_tests.sh`
**Functions:** `run_t3000_llama3_tests`, `run_t3000_llama3_70b_tests`, `run_t3000_llama3_90b_tests`

```
pytest models/tt_transformers/tests/test_model_prefill.py
pytest models/tt_transformers/tests/test_chunked_generation.py
```

---

## models/tt_transformers/tests/mixtral/test_mixtral_model_prefill.py

**CI Workflow:** `t3000-unit-tests-impl.yaml`
**Shell script:** `tests/scripts/t3000/run_t3000_unit_tests.sh` → `run_t3000_mixtral_tests`
**Job name:** `t3k_mixtral_tests`

```
HF_MODEL=$mixtral8x7 TT_CACHE_PATH=$tt_cache_mixtral8x7 CI=true \
pytest models/tt_transformers/tests/mixtral/test_mixtral_model_prefill.py::test_model_inference[wormhole_b0-device_params0-1layer-performance-max128k-4k-page_params0-paged_attention-8] --timeout=720
```

---

## models/experimental/gemma3_4b/tests/vision_tests/test_end2end.py
## models/experimental/mistral_24b/tests/pipeline_tests/test_end2end.py

**CI Workflow:** NOT IN ANY CI PIPELINE
**Note:** These files are new and have no CI pipeline references anywhere in
.github/workflows/ or tests/pipeline_reorg/ or tests/scripts/. They need to be
run manually or added to a pipeline before merging.

---

## Summary Table

| Test / Job | Files Covered | In Pipeline? |
|---|---|---|
| `models_tttv2_llama31_8B_tests` | `common/demos/llama31_8B_demo.py` | ✅ |
| `t3k_gpt_oss_tests` + Galaxy GPT-OSS | `demos/gpt_oss/demo/text_demo.py`<br>`demos/gpt_oss/tests/accuracy/test_model.py` | ✅ |
| Galaxy llama3_70b tests | `demos/llama3_70b_galaxy/demo/text_demo.py`<br>`demos/llama3_70b_galaxy/demo/text_qwen_demo.py` | ✅ |
| `t3k_gemma3_tests` + N300 single-card | `demos/multimodal/gemma3/demo/text_demo.py`<br>`demos/multimodal/gemma3/demo/vision_demo.py` | ✅ |
| `t3k_qwen25_vl_tests` + N150/N300 single-card | `demos/qwen25_vl/demo/demo.py` | ✅ |
| `t3k_qwen3_vl_tests` | `demos/qwen3_vl/demo/demo.py` | ✅ |
| Galaxy demo (placeholder only) | `demos/tg/llama3_70b/demo/demo.py` | ✅ (placeholder only) |
| t3k/single-card/galaxy demo tests (various) | `tt_transformers/demo/simple_text_demo.py` | ✅ |
| `t3k_llama3_vision_tests` / `t3k_llama3_90b_vision_tests` | `tt_transformers/demo/simple_vision_demo.py` | ✅ |
| t3000-integration tests | `tt_transformers/tests/test_model_prefill.py`<br>`tt_transformers/tests/test_chunked_generation.py` | ✅ |
| `t3k_mixtral_tests` | `tt_transformers/tests/mixtral/test_mixtral_model_prefill.py` | ✅ |
| ❌ Not in any pipeline | `tt_transformers/demo/multimodal_demo_chat.py`<br>`tt_transformers/demo/multimodal_demo_text.py` | ❌ |
| ❌ Not in any pipeline | `experimental/gemma3_4b/tests/vision_tests/test_end2end.py`<br>`experimental/mistral_24b/tests/pipeline_tests/test_end2end.py` | ❌ |
