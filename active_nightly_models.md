# Active Nightly CI Models — Tracing Targets

This document enumerates every model that is **actively executed on a schedule** in the tt-metal models CI, broken down by architecture/SKU. For each entry it gives the exact pytest invocation and the upstream requirements (env vars, weight paths, pip installs) needed to reproduce the run locally.

Sources (all paths relative to the repo root):
- [.github/workflows/models-t1-unit-tests.yaml](.github/workflows/models-t1-unit-tests.yaml), [models-t2-unit-tests.yaml](.github/workflows/models-t2-unit-tests.yaml), [models-t3-unit-tests.yaml](.github/workflows/models-t3-unit-tests.yaml)
- [.github/workflows/models-t1-e2e-tests.yaml](.github/workflows/models-t1-e2e-tests.yaml), [models-t2-e2e-tests.yaml](.github/workflows/models-t2-e2e-tests.yaml), [models-t3-e2e-tests.yaml](.github/workflows/models-t3-e2e-tests.yaml)
- [.github/workflows/galaxy-demo-tests.yaml](.github/workflows/galaxy-demo-tests.yaml), [t3000-demo-tests.yaml](.github/workflows/t3000-demo-tests.yaml), [blackhole-demo-tests.yaml](.github/workflows/blackhole-demo-tests.yaml)
- [tests/pipeline_reorg/models_unit_tests.yaml](tests/pipeline_reorg/models_unit_tests.yaml), [models_e2e_tests.yaml](tests/pipeline_reorg/models_e2e_tests.yaml)
- [tests/pipeline_reorg/galaxy_demo_tests.yaml](tests/pipeline_reorg/galaxy_demo_tests.yaml), [t3k_demo_tests.yaml](tests/pipeline_reorg/t3k_demo_tests.yaml), [blackhole_demo_tests.yaml](tests/pipeline_reorg/blackhole_demo_tests.yaml)
- [tests/scripts/t3000/run_t3000_demo_tests.sh](tests/scripts/t3000/run_t3000_demo_tests.sh)

Schedule cadence:
- **galaxy-demo-tests** — daily 00:00 UTC
- **t3000-demo-tests** — Mon/Wed/Fri 00:00 UTC
- **blackhole-demo-tests** — daily 04:00 UTC
- **models-t1/t2/t3-unit-tests** — multiple times daily
- **models-t1/t2/t3-e2e-tests** — daily

Common assumptions (apply to every entry unless overridden):
- `cd $TT_METAL_HOME && export PYTHONPATH=$TT_METAL_HOME` before pytest.
- `ARCH_NAME` is set per the device (`wormhole_b0` or `blackhole`).
- Caches live under `/mnt/MLPerf/huggingface/tt_cache/...` for WH (HF mount) or `/localdev/blackhole_demos/huggingface_data/...` for BH.
- HF weights are pre-staged on the runners; offline mode is used where applicable (`HF_HUB_OFFLINE=1`).
- `CI=true` is exported for tests that gate behavior on it (e.g. Gemma-4 mesh selection).

### Local-machine env overrides (this Galaxy box)

`/mnt/MLPerf` is not mounted on this host. Use these substitutions when running locally:

| CI path | Local equivalent |
|---|---|
| `/mnt/MLPerf/huggingface/tt_cache/...` | `/data/MLPerf/huggingface/tt_cache/...` (only `openai--gpt-oss-120b` populated) |
| `/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked` | `/data/deepseek/DeepSeek-R1-0528-dequantized-stacked` |
| `/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI` | `/data/deepseek/DeepSeek-R1-0528-Cache/CI` |
| `/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/` | `/home/ubuntu/Resources/Llama3.3-70B-Instruct/` (writable) |

HF auth + cache live under `/home/stevenlee/ml_cache/hf/`:
```
export HF_HOME=/home/stevenlee/ml_cache/hf
export HF_TOKEN=$(cat /home/stevenlee/ml_cache/hf/token)
```
This grants access to `black-forest-labs/FLUX.1-dev` (already cached locally). It does **not** grant access to `stabilityai/stable-diffusion-3.5-large` (sd35 demo).

Other quirks observed locally:
- `python_env` has `transformers 4.57.0` but project pins `==4.53.0` → sentence_bert tg test hits `KeyError: None` from a HF default change. Pin back to fix.
- `models/demos/llama3_70b_galaxy/demo/text_demo.py` torch.load fails on the consolidated.0N.pth files even with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` (UnpicklingError "invalid load key"). Needs a different weights snapshot or test-side fix.

---

## Wormhole — wh_n150 (single-card N150)

### llama3.1-8b — unit (tier 1) + e2e (tier 1)
**Env**
```
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.1-8B-Instruct
```
**Unit tests**
```
pytest --timeout 300 models/tt_transformers/tests/test_embedding.py
pytest --timeout 300 models/tt_transformers/tests/test_rms_norm.py
pytest --timeout 600 models/tt_transformers/tests/test_mlp.py
pytest --timeout 300 models/tt_transformers/tests/test_attention.py
pytest --timeout 300 models/tt_transformers/tests/test_attention_prefill.py
pytest --timeout 300 models/tt_transformers/tests/test_decoder.py
pytest --timeout 400 models/tt_transformers/tests/test_decoder_prefill.py
```
**E2E**
```
pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
pytest --timeout 600 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"
```

### llama3.2-1b — unit (tier 3) + e2e (tier 3)
**Env**
```
export HF_MODEL=meta-llama/Llama-3.2-1B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.2-1B-Instruct
```
**Unit**
```
pytest --timeout 300 models/tt_transformers/tests/test_decoder.py -k 32-page_params
pytest --timeout 300 models/tt_transformers/tests/test_decoder_prefill.py -k 128-page_params
```
**E2E**
```
pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
```

### llama3.2-3b — unit (tier 3) + e2e (tier 3)
**Env**
```
export HF_MODEL=meta-llama/Llama-3.2-3B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.2-3B-Instruct
```
**Unit**
```
pytest --timeout 300 models/tt_transformers/tests/test_decoder.py -k 32-page_params
pytest --timeout 300 models/tt_transformers/tests/test_decoder_prefill.py -k 128-page_params
```
**E2E**
```
pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
```

### mistral-7b — unit (tier 2) + e2e (tier 2)
**Env**
```
export HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/mistralai/Mistral-7B-Instruct-v0.3
```
**Unit**
```
pytest --timeout 600 models/tt_transformers/tests/test_embedding.py
pytest --timeout 600 models/tt_transformers/tests/test_attention.py
pytest --timeout 600 models/tt_transformers/tests/test_attention_prefill.py
pytest --timeout 600 models/tt_transformers/tests/test_mlp.py
pytest --timeout 600 models/tt_transformers/tests/test_rms_norm.py
pytest --timeout 600 models/tt_transformers/tests/test_decoder.py
pytest --timeout 600 models/tt_transformers/tests/test_decoder_prefill.py
```
**E2E**
```
pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
pytest --timeout 1800 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"
```

### gemma-3-4b — unit (tier 2) + e2e (tier 2)
**Env**
```
export HF_MODEL=google/gemma-3-4b-it
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/google/gemma-3-4b-it
```
**Unit**
```
pytest --timeout 600 models/demos/multimodal/gemma3/tests/test_ci_dispatch.py -k "4b"
```
**E2E**
```
pytest --timeout 1000 models/demos/multimodal/gemma3/demo/text_demo.py -k "performance and ci-1"
pytest --timeout 1000 models/demos/multimodal/gemma3/demo/vision_demo.py -k "performance and batch1-multi-image-trace"
```

### gemma-4-e2b — unit (tier 2) + e2e (tier 2)
**Pip install**
```
uv pip install -r models/demos/gemma4/requirements.txt
```
**Env**
```
export HF_HUB_OFFLINE=1 HF_HOME=/mnt/MLPerf/huggingface
export HF_MODEL=google/gemma-4-E2B-it
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/google--gemma-4-E2B-it
export CI=true
```
**Unit**
```
pytest --timeout 600 models/demos/gemma4/tests/unit/
```
**E2E**
```
pytest -s --timeout 1000 models/demos/gemma4/demo/text_demo.py::test_demo
```
> Note: Gemma-4 entries use `parametrize_mesh_with_fabric()` — the same invocation auto-selects 1x1 mesh on N150 when `CI=true`.

### gemma-4-e4b — unit (tier 2) + e2e (tier 2)
Same as gemma-4-e2b, but `HF_MODEL=google/gemma-4-E4B-it` and `TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/google--gemma-4-E4B-it`.

### whisper — unit (tier 1) + e2e (tier 1)
**Env**
```
export HF_HUB_OFFLINE=0 HF_HOME=/mnt/MLPerf/huggingface
export HF_HUB_CACHE=/mnt/MLPerf/huggingface/hub
export HF_DATASETS_CACHE=/tmp/huggingface/datasets
```
**Unit**
```
pytest --timeout 600 models/demos/audio/whisper/tests/test_whisper_modules.py
```
**E2E**
```
pytest --timeout=600 models/demos/audio/whisper/demo/demo.py --input-path="models/demos/audio/whisper/demo/dataset/conditional_generation" -k "conditional_generation"
```

### falcon-7b — unit (tier 3) + e2e (tier 3)
**Unit**
```
pytest --timeout 300 models/demos/ttnn_falcon7b/tests/test_falcon_decoder.py
```
**E2E**
```
pytest --timeout 420 --disable-warnings -q -s --input-method=cli --cli-input="YOUR PROMPT GOES HERE!" models/demos/wormhole/falcon7b/demo_wormhole.py::test_demo -k "default_mode_1024_stochastic"
```

### phi-3-mini — unit (tier 3) + e2e (tier 3)
**Env**
```
export HF_MODEL=microsoft/Phi-3-mini-128k-instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/microsoft/Phi-3-mini-128k-instruct
```
**Unit**
```
pytest --timeout 180 models/tt_transformers/tests/test_decoder.py -k 32-page_params
pytest --timeout 180 models/tt_transformers/tests/test_decoder_prefill.py -k 128-page_params
```
**E2E**
```
pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
```

### shallow-unet — unit (tier 2)
```
pytest --timeout 600 models/experimental/functional_unet/tests/test_unet_model.py
```

---

## Wormhole — wh_n300 (dual-card N300)

### qwen2.5-7b — unit (tier 3) + e2e (tier 3)
**Env**
```
export HF_MODEL=Qwen/Qwen2.5-7B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen2.5-7B-Instruct
```
**Unit**
```
pytest --timeout 300 models/tt_transformers/tests/test_decoder.py -k 32-page_params
pytest --timeout 300 models/tt_transformers/tests/test_decoder_prefill.py -k 128-page_params
```
**E2E**
```
pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
```

> Note: many other models that previously ran on n300 are commented out due to reduced n300 runner capacity (gemma-4-e4b, shallow-unet, whisper, mamba-2.8b, phi-4-mini).

---

## Wormhole — wh_llmbox (LLMBox 4-chip)

### gemma-4-26b-a4b — unit (tier 2)
**Pip install**
```
uv pip install -r models/demos/gemma4/requirements.txt
```
**Env**
```
export HF_HUB_OFFLINE=1 HF_HOME=/mnt/MLPerf/huggingface
export HF_MODEL=google/gemma-4-26B-A4B-it
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/google--gemma-4-26B-A4B-it
export CI=true
```
```
pytest --timeout 900 models/demos/gemma4/tests/unit/
```

### gemma-4-31b — unit (tier 2)
Same as above with `HF_MODEL=google/gemma-4-31B-it` and `TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/google--gemma-4-31B-it`.

---

## Wormhole — wh_llmbox_perf (T3K perf, 8-chip)

### llama3.1-8b — unit (tier 2) + e2e (tier 2)
Same commands and env as on **wh_n150** (above). Mesh selection handled by the test config.

### llama3.1-8b-dp (data-parallel) — e2e (tier 2)
**Env**
```
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.1-8B-Instruct
```
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-b1-DP-4 or performance-ci-b1-DP-8" --timeout 1000
```

### llama3.2-11b-vision — unit (tier 3) + e2e (tier 3)
**Env**
```
export HF_MODEL=meta-llama/Llama-3.2-11B-Vision-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.2-11B-Vision-Instruct
export MESH_DEVICE=T3K   # e2e only
```
**Unit**
```
pytest --timeout 300 models/tt_transformers/tests/test_decoder.py -k 32-page_params
pytest --timeout 300 models/tt_transformers/tests/test_decoder_prefill.py -k 128-page_params
```
**E2E**
```
pytest --timeout 900 models/tt_transformers/demo/simple_vision_demo.py -k "not batch1-notrace"
```

### llama3.2-90b-vision — unit (tier 2) + e2e (tier 2)
**Env**
```
export HF_MODEL=meta-llama/Llama-3.2-90B-Vision-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.2-90B-Vision-Instruct
export MESH_DEVICE=T3K
```
**Unit**
```
pytest --timeout 900 models/tt_transformers/tests/test_decoder.py
pytest --timeout 900 models/tt_transformers/tests/test_decoder_prefill.py
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_image_mlp.py
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_image_attention.py
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_image_block.py
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_cross_attention.py -k "batch_1"
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_cross_block.py -k "batch_1"
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_conv2d_patch.py
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_class_embedding.py
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_tile_position_embedding.py
pytest --timeout 900 models/tt_transformers/tests/multimodal/test_llama_positional_embedding.py
```
**E2E**
```
pytest --timeout 1000 models/tt_transformers/demo/simple_vision_demo.py -k "batch1-trace"
```

### llama3.3-70b — unit (tier 2) + e2e (tier 2)
**Env**
```
export HF_MODEL=meta-llama/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama/Llama-3.3-70B-Instruct
```
**Unit**
```
pytest --timeout 900 models/tt_transformers/tests/test_embedding.py
pytest --timeout 900 models/tt_transformers/tests/test_rms_norm.py
pytest --timeout 900 models/tt_transformers/tests/test_mlp.py
pytest --timeout 900 models/tt_transformers/tests/test_attention.py
pytest --timeout 900 models/tt_transformers/tests/test_attention_prefill.py
pytest --timeout 900 models/tt_transformers/tests/test_decoder.py
pytest --timeout 900 models/tt_transformers/tests/test_decoder_prefill.py
```
**E2E**
```
pytest --timeout 4200 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
pytest --timeout 1800 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"
```

### llama3.3-70b-dp — e2e (tier 2)
Same env as llama3.3-70b.
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-b1-DP-4 or performance-ci-b1-DP-8" --timeout 1000
```

### qwen3-32b — unit (tier 2) + e2e (tier 2)
**Env**
```
export HF_MODEL=Qwen/Qwen3-32B
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B
```
**Unit**
```
pytest --timeout 900 models/tt_transformers/tests/test_attention.py
pytest --timeout 900 models/tt_transformers/tests/test_attention_prefill.py
pytest --timeout 900 models/tt_transformers/tests/test_mlp.py
pytest --timeout 900 models/tt_transformers/tests/test_rms_norm.py
pytest --timeout 900 models/tt_transformers/tests/test_decoder.py
pytest --timeout 900 models/tt_transformers/tests/test_decoder_prefill.py
```
**E2E**
```
pytest --timeout 4200 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
pytest --timeout 1800 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"
```

### qwen2.5-32b — unit (tier 2) + e2e (tier 2)
**Env**
```
export HF_MODEL=Qwen/Qwen2.5-32B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen2.5-32B-Instruct
```
**Unit**
```
pytest --timeout 900 models/tt_transformers/tests/test_attention.py
pytest --timeout 900 models/tt_transformers/tests/test_attention_prefill.py
pytest --timeout 900 models/tt_transformers/tests/test_mlp.py
pytest --timeout 900 models/tt_transformers/tests/test_rms_norm.py
pytest --timeout 900 models/tt_transformers/tests/test_decoder.py
pytest --timeout 900 models/tt_transformers/tests/test_decoder_prefill.py
```
**E2E**
```
pytest --timeout 420 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
pytest --timeout 1800 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"
```

### qwen2.5-coder-32b — unit (tier 2) + e2e (tier 2)
Same as qwen2.5-32b but `HF_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct` / `TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen2.5-Coder-32B-Instruct`.

### qwen2.5-72b — unit (tier 3)
**Env**
```
export HF_MODEL=Qwen/Qwen2.5-72B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen2.5-72B-Instruct
```
```
pytest --timeout 300 models/tt_transformers/tests/test_decoder.py -k 32-page_params
pytest --timeout 300 models/tt_transformers/tests/test_decoder_prefill.py -k 128-page_params
```
> e2e currently disabled (ci-token-matching hangs).

### qwen2.5-72b-vl — unit (tier 2) + e2e (tier 2)
**Pip install**
```
uv pip install -r models/demos/qwen25_vl/requirements.txt
```
**Env**
```
export HF_MODEL=Qwen/Qwen2.5-VL-72B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen2.5-VL-72B-Instruct
export MESH_DEVICE=T3K
```
**Unit**
```
pytest --timeout 600 models/demos/qwen25_vl/tests/ --ignore=models/demos/qwen25_vl/tests/test_ci_dispatch.py --ignore=models/demos/qwen25_vl/tests/conftest.py
```
**E2E**
```
pytest --timeout 900 models/demos/qwen25_vl/demo/demo.py
```

### qwen2.5-vl-32b — unit (tier 3) + e2e (tier 3)
**Pip install**
```
uv pip install -r models/demos/qwen25_vl/requirements.txt
```
**Env**
```
export HF_MODEL=Qwen/Qwen2.5-VL-32B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen2.5-VL-32B-Instruct
export MESH_DEVICE=T3K
```
**Unit**
```
pytest --timeout 300 models/demos/qwen25_vl/tests/test_model.py
```
**E2E**
```
pytest --timeout 600 models/demos/qwen25_vl/demo/demo.py
```

### qwq-32b — unit (tier 3) + e2e (tier 3)
**Env**
```
export HF_MODEL=Qwen/QwQ-32B
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/QwQ-32B
```
**Unit**
```
pytest --timeout 300 models/tt_transformers/tests/test_decoder.py
pytest --timeout 300 models/tt_transformers/tests/test_decoder_prefill.py
```
**E2E**
```
pytest --timeout 600 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"
```

### gemma-3-27b — unit (tier 2) + e2e (tier 2)
**Env**
```
export HF_MODEL=google/gemma-3-27b-it
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/google/gemma-3-27b-it
```
**Unit**
```
pytest --timeout 600 models/demos/multimodal/gemma3/tests/test_ci_dispatch.py -k "27b"
```
**E2E**
```
pytest --timeout 1000 models/demos/multimodal/gemma3/demo/text_demo.py -k "performance and ci-1"
pytest --timeout 1000 models/demos/multimodal/gemma3/demo/vision_demo.py -k "performance and batch1-multi-image-trace"
```

### gemma-4-26b-a4b — e2e (tier 2)
Env identical to wh_llmbox unit entry above. Demo:
```
pytest -s --timeout 1500 models/demos/gemma4/demo/text_demo.py::test_demo
```

### gemma-4-31b — e2e (tier 2)
Same with `HF_MODEL=google/gemma-4-31B-it` / `TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/google--gemma-4-31B-it`.

### mixtral-8x7b — unit (tier 2) + e2e (tier 2)
**Unit env**
```
export HF_MODEL=mistralai/Mixtral-8x7B-v0.1
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/mistralai/Mixtral-8x7B-v0.1
```
**Unit**
```
pytest --timeout 120 models/tt_transformers/tests/mixtral/test_mixtral_rms_norm.py
pytest --timeout 120 models/tt_transformers/tests/mixtral/test_mixtral_mlp.py
pytest --timeout 120 models/tt_transformers/tests/mixtral/test_mixtral_moe.py
pytest --timeout 120 models/tt_transformers/tests/mixtral/test_mixtral_decoder.py
pytest --timeout 120 models/tt_transformers/tests/mixtral/test_mixtral_decoder_prefill.py
pytest --timeout 120 models/tt_transformers/tests/mixtral/test_mixtral_model.py -k "performance and not accuracy and quick"
pytest --timeout 120 models/tt_transformers/tests/mixtral/test_mixtral_model_prefill.py -k "paged_attention"
```
**E2E env** (instruct variant)
```
export HF_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/mistralai/Mixtral-8x7B-Instruct-v0.1
```
**E2E**
```
pytest --timeout 4200 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-token-matching"
pytest --timeout 1800 models/tt_transformers/demo/simple_text_demo.py -k "performance-ci-eval-32"
```

### gpt-oss-20b — e2e (tier 2)
**Pip install**
```
uv pip install -r models/demos/gpt_oss/requirements.txt
```
**Env**
```
export HF_MODEL=openai/gpt-oss-20b
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/openai--gpt-oss-20b/
```
```
pytest models/demos/gpt_oss/demo/text_demo.py -k "1x8" --timeout 600
```
> Unit tests for gpt-oss-20b are currently disabled (segfault on CI).

### falcon-40b — unit (tier 3) + e2e (tier 3)
**Unit**
```
pytest --timeout 300 models/demos/t3000/falcon40b/tests/test_falcon_decoder.py
```
**E2E**
```
pytest --timeout 720 models/demos/t3000/falcon40b/tests/test_demo.py::test_demo[wormhole_b0-mesh_device0-True-device_params0-128]
```

### shallow-unet — unit (tier 2)
```
pytest --timeout 600 models/experimental/functional_unet/tests/test_unet_model.py
```

### T3K demos (M/W/F via t3000-demo-tests)
The pipeline_reorg yaml dispatches via shell functions in [tests/scripts/t3000/run_t3000_demo_tests.sh](tests/scripts/t3000/run_t3000_demo_tests.sh). Concrete pytest invocations:

#### resnet50 (t3k_resnet50_tests)
```
pytest models/demos/vision/classification/resnet50/ttnn_resnet/tests/test_demo.py --timeout=720
```

#### sentence_bert (t3k_sentence_bert_tests)
```
pytest models/demos/t3000/sentence_bert/demo/demo.py --timeout=600
```

#### sd35_large (t3k_sd35_large_tests)
**Env**: `NO_PROMPT=1`
```
pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k "2x4cfg1sp0tp1" --timeout 1200
```

#### flux1-dev (t3k_flux1-dev_tests)
**Env**: `NO_PROMPT=1`
```
pytest models/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "2x4sp0tp1-dev" --timeout 1200
```

#### motif (t3k_motif_tests)
**Env**: `NO_PROMPT=1`
```
pytest models/tt_dit/tests/models/motif/test_pipeline_motif.py -k "2x4cfg0sp0tp1" --timeout 1200
```

#### qwenimage (t3k_qwenimage_tests)
**Env**: `NO_PROMPT=1`
```
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py -k "2x4" --timeout 1200
```

#### qwen3_vl (t3k_qwen3_vl_tests)
**Pip install**
```
uv pip install -r models/demos/qwen3_vl/requirements.txt
```
**Env**
```
export PYTEST_ADDOPTS="--tb=short"
export MESH_DEVICE=T3K
export HF_MODEL=Qwen/Qwen3-VL-32B-Instruct
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/Qwen/Qwen3-VL-32B-Instruct
```
```
pytest models/demos/qwen3_vl/demo/demo.py --timeout 600
```

#### wan22 (t3k_wan2.2_tests)
**Env**
```
export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
export NO_PROMPT=1
```
```
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "2x4sp0tp1 and resolution_480p" --timeout 1500
```

#### mochi (t3k_mochi_tests)
**Env**
```
export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
```
```
pytest models/tt_dit/tests/models/mochi/test_pipeline_mochi.py -k "dit_2x4sp0tp1_vae_1x8sp0tp1" --timeout 1500
```

---

## Wormhole — wh_galaxy_perf (TG, 32-chip)

### qwen3-32b-galaxy — unit (tier 1) + e2e (tier 1)
**Env**
```
export HF_MODEL=Qwen/Qwen3-32B
export TT_CACHE_PATH=/data/MLPerf/huggingface/tt_cache/Qwen/Qwen3-32B
```
**Unit**
```
pytest --timeout 900 models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_mlp.py
pytest --timeout 900 models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_mlp_prefill.py
pytest --timeout 900 models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_attention.py
pytest --timeout 900 models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_attention_prefill.py
pytest --timeout 900 models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_decoder.py
pytest --timeout 900 models/demos/llama3_70b_galaxy/tests/unit_tests/test_qwen_decoder_prefill.py
```
**E2E**
```
pytest --timeout 1000 models/demos/llama3_70b_galaxy/tests/test_qwen_accuracy.py
```

### gpt-oss-120b-high-batch-galaxy — unit (tier 1) + e2e (tier 1)
**Pip install**
```
uv pip install -r models/demos/gpt_oss/requirements.txt
```
**Env**
```
export HF_MODEL=openai/gpt-oss-120b
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/openai--gpt-oss-120b/
```
**Unit**
```
pytest --timeout 1500 models/demos/gpt_oss/tests/unit -k "4x8 and decode_high_throughput"
```
**E2E**
```
pytest models/demos/gpt_oss/demo/text_demo.py -k "mesh_4x8 and batch128" --timeout 1200
```

### llama3.3-70b-galaxy — e2e (tier 1)
**Env**
```
export TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache
export LLAMA_DIR=/mnt/MLPerf/tt_dnn-models/llama/Llama3.3-70B-Instruct/
export FAKE_DEVICE=TG
```
```
pytest --timeout 1000 models/demos/llama3_70b_galaxy/demo/text_demo.py -k "pcc-80L"
pytest --timeout 1000 models/demos/llama3_70b_galaxy/demo/text_demo.py -k "repeat2"
```
> Unit tests disabled (issue #42139). DP variants disabled (issue #42553).

### Galaxy demos (daily via galaxy-demo-tests)

#### sentence_bert (Galaxy)
```
pytest models/demos/tg/sentence_bert/tests/test_sentence_bert_e2e_performant.py --timeout=1500
```

#### sd35
**Env**
```
export TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache
export NO_PROMPT=1
export TT_MM_THROTTLE_PERF=5
```
```
pytest models/tt_dit/tests/models/sd35/test_pipeline_sd35.py -k "4x8cfg1sp0tp1" --timeout 1200
```

#### flux1
**Env**
```
export TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache
export NO_PROMPT=1
export TT_MM_THROTTLE_PERF=5
```
```
pytest models/tt_dit/tests/models/flux1/test_pipeline_flux1.py -k "4x8sp0tp1-dev" --timeout 1200
```

#### motif (Galaxy)
**Env**
```
export TT_CACHE_HOME=/mnt/MLPerf/huggingface/tt_cache
export NO_PROMPT=1
```
```
pytest models/tt_dit/tests/models/motif/test_pipeline_motif.py -k "4x8cfg1sp0tp1" --timeout 1200
```

#### wan22 (Galaxy)
**Env**
```
export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
export NO_PROMPT=1
```
```
pytest models/tt_dit/tests/models/wan2_2/test_pipeline_wan.py -k "wh_4x8sp1tp0 and resolution_720p" --timeout 1500
```

#### mochi (Galaxy)
**Env**
```
export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
export NO_PROMPT=1
```
```
pytest models/tt_dit/tests/models/mochi/test_pipeline_mochi.py -k "wh_4x8sp1tp0" --timeout=1500
```

#### deepseek_v3
**Env**
```
export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked
export DEEPSEEK_V3_CACHE=/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI
export MESH_DEVICE=TG
```
```
pytest models/demos/deepseek_v3/demo/test_demo.py -k "tg_stress or tg_upr8" --timeout 900
```

#### qwenimage (Galaxy)
**Env**
```
export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
export NO_PROMPT=1
```
```
pytest models/tt_dit/tests/models/qwenimage/test_pipeline_qwenimage.py -k "4x8" --timeout 1200
```

---

## Blackhole — bh_p150 / bh_p150_perf (single P150)

All Blackhole demos run daily via [blackhole-demo-tests.yaml](.github/workflows/blackhole-demo-tests.yaml). Weights cache mode varies per SKU (local-disk / cloud-mlperf / lfc).

### whisper (bh_p150_perf — performance)
```
pytest models/demos/audio/whisper/demo/demo.py --input-path="models/demos/audio/whisper/demo/dataset/conditional_generation" -k "conditional_generation" --timeout=600
```

### whisper (bh_p150 — nightly modules)
```
pytest models/demos/audio/whisper/tests/test_whisper_modules.py --timeout=600
```

### llama3.1-8b (bh_p150_perf — performance)
**Env**
```
export HF_MODEL=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.1-8B-Instruct
```
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci and not stress" --max_seq_len 131072
```

### llama3.1-8b (bh_p150 — stress)
Same env.
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and stress" --timeout=1800
```

### unet-shallow (bh_p150_perf)
```
pytest -sv models/experimental/functional_unet/tests/test_unet_perf.py -k "test_unet_trace_perf and not test_unet_trace_perf_multi_device"
```

### stable_diffusion_xl (bh_p150b_civ2)
```
pytest models/demos/stable_diffusion_xl_base/demo/demo.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
pytest models/demos/stable_diffusion_xl_base/demo/demo_base_and_refiner.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
pytest models/demos/stable_diffusion_xl_base/demo/demo_img2img.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
pytest models/demos/stable_diffusion_xl_base/demo/demo_inpainting.py -k "device_vae and device_encoders and with_trace and no_cfg_parallel"
```

---

## Blackhole — bh_p300 (1xP300, viommu)

### llama3.1-8b — performance batch-32 (also bh_llmbox/bh_deskbox/bh_loudbox/bh_quietbox_2)
**Env**
```
export HF_MODEL=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.1-8B-Instruct
export TT_CACHE_PATH=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.1-8B-Instruct
```
```
pytest --timeout 600 models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --data_parallel 1 --max_seq_len 131072 --use_prefetcher True
```

### llama3.1-8b — performance batch-1 (P300/llmbox/deskbox/loudbox/quietbox_2)
Same env.
```
pytest --timeout 600 models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-1" --data_parallel 1 --max_seq_len 131072 --use_prefetcher True
```

### llama3.1-8b — DP=2 performance (P300, deskbox)
Same env.
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --data_parallel 2 --max_seq_len 131072 --timeout=600
```

### llama3.1-8b — DP=2 stress (P300, deskbox)
Same env.
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and stress" --data_parallel 2 --max_generated_tokens 22000 --timeout=3600
```

### llama3.1-8b — TP device perf decode/prefill (P300/llmbox/deskbox/loudbox/quietbox_2)
**Env**
```
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
export HF_MODEL=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.1-8B-Instruct
```
```
pytest --timeout 600 models/tt_transformers/tests/test_device_perf.py -k "decode-llama3_8b-2-131072-2-10-1-1-False"
pytest --timeout 600 models/tt_transformers/tests/test_device_perf.py -k "prefill-llama3_8b-2-131072-2-2-1-1-False"
```

### llama3-8b — performance batch-32 (P300/llmbox/deskbox/loudbox/quietbox_2)
**Env**
```
export HF_MODEL=models/tt_transformers/model_params/Meta-Llama-3-8B
export TT_CACHE_PATH=models/tt_transformers/model_params/Meta-Llama-3-8B
```
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --max_seq_len 131072 --timeout=600
```

### flux.1-dev (P300, viommu — 1x2sp0tp1)
**Env**: `HF_HUB_CACHE=/localdev/blackhole_demos/huggingface_data/black-forest-labs`
```
pytest models/tt_dit/tests/models/flux1/test_performance_flux1.py -k "1x2sp0tp1" --timeout=600
```

---

## Blackhole — bh_deskbox (2xP150)

Already covered in P300 section above (shared rows): llama3.1-8b batch-32, batch-1, DP=2 perf, DP=2 stress, TP device perf decode/prefill, llama3-8b batch-32.

---

## Blackhole — bh_llmbox (Quietbox 4xP150)

Inherits llama3.1-8b batch-32, batch-1, TP device perf decode/prefill, llama3-8b batch-32 (same commands as above), plus:

### llama3.1-8b — DP=4 performance (llmbox, quietbox_2)
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --data_parallel 4 --max_seq_len 131072 --timeout=600
```

### llama3.1-8b — DP=4 stress (llmbox, quietbox_2)
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and stress" --data_parallel 4 --max_generated_tokens 22000 --timeout=3600
```

### llama3.3-70b — performance batch-32 (llmbox, loudbox, quietbox_2)
**Env**
```
export HF_MODEL=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.3-70B-Instruct
export TT_CACHE_PATH=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.3-70B-Instruct
```
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --max_seq_len 131072 --timeout=600
```

### llama3.3-70b — TP device perf decode/prefill (llmbox, loudbox, quietbox_2)
**Env**
```
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
export HF_MODEL=/localdev/blackhole_demos/huggingface_data/meta-llama/Llama-3.3-70B-Instruct
```
```
pytest --timeout 900 models/tt_transformers/tests/test_device_perf.py -k "prefill-llama3_70b-2-131072-2-2-1-1-False"
pytest --timeout 900 models/tt_transformers/tests/test_device_perf.py -k "decode-llama3_70b-2-131072-2-10-1-1-False"
```

### qwen3-32b — performance batch-32 (llmbox, loudbox, quietbox_2)
**Env**
```
export HF_MODEL=/localdev/blackhole_demos/huggingface_data/Qwen/Qwen3-32B
export TT_CACHE_PATH=/localdev/blackhole_demos/huggingface_data/Qwen/Qwen3-32B
```
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --max_seq_len 32768 --timeout=600
```

### qwen3-32b — TP device perf decode/prefill (llmbox, loudbox, quietbox_2)
**Env**
```
export TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=10000
export HF_MODEL=/localdev/blackhole_demos/huggingface_data/Qwen/Qwen3-32B
```
```
pytest --timeout 900 models/tt_transformers/tests/test_device_perf.py -k "prefill-qwen3_32b-2-32768-2-2-1-1-False"
pytest --timeout 900 models/tt_transformers/tests/test_device_perf.py -k "decode-qwen3_32b-2-32768-2-10-1-1-False"
```

### qwen2.5-32b-instruct — performance batch-32 (llmbox, loudbox, quietbox_2)
**Env**
```
export HF_MODEL=models/tt_transformers/model_params/Qwen2.5-32B-Instruct
export TT_CACHE_PATH=models/tt_transformers/model_params/Qwen2.5-32B-Instruct
```
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --max_seq_len 131072 --timeout=600
```

### qwen2.5-72b-instruct — performance batch-32 (llmbox, loudbox, quietbox_2)
Same with `Qwen2.5-72B-Instruct` paths.

### qwen2.5-vl-32b-instruct — performance batch-32 (llmbox, loudbox, quietbox_2)
**Pip install**
```
uv pip install -r models/demos/qwen25_vl/requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```
**Env** (note: `$qwen25_vl_32b` and `$tt_cache_32b` are referenced from CI shell scope — for local repro substitute the Qwen2.5-VL-32B-Instruct paths)
```
export HF_MODEL=Qwen/Qwen2.5-VL-32B-Instruct
export TT_CACHE_PATH=/localdev/blackhole_demos/huggingface_data/Qwen/Qwen2.5-VL-32B-Instruct
export MESH_DEVICE=T3K
export CI=true
```
```
pytest --timeout 600 models/demos/qwen25_vl/demo/demo.py
```

### qwen2.5-vl-72b — performance batch-32 (llmbox, loudbox, quietbox_2)
**Pip install**
```
uv pip install --system -r models/demos/qwen25_vl/requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```
**Env**: `HF_HUB_CACHE=/localdev/blackhole_demos/huggingface_data/Qwen`
```
pytest --timeout 600 models/demos/qwen25_vl/demo/demo.py
```

### qwen3-vl-32b — performance batch-32 (llmbox, loudbox, quietbox_2)
**Env**
```
export HF_MODEL=models/tt_transformers/model_params/Qwen3-VL-32B-Instruct
export TT_CACHE_PATH=models/tt_transformers/model_params/Qwen3-VL-32B-Instruct
```
```
pytest models/demos/qwen3_vl/tests/ --ignore=models/demos/qwen3_vl/tests/test_ci_dispatch.py -v
```

### mistral-small-3.1-24b — pipeline_tests (llmbox, loudbox, quietbox_2)
**Env**
```
export HF_MODEL=models/tt_transformers/model_params/Mistral-Small-3.1-24B-Instruct-2503
export TT_CACHE_PATH=models/tt_transformers/model_params/Mistral-Small-3.1-24B-Instruct-2503
export MESH_DEVICE=T3K
```
```
pytest --timeout 600 models/experimental/mistral_24b/tests/pipeline_tests
```

### gpt-oss-20b — performance (llmbox, loudbox)
**Env**
```
export HF_MODEL=openai/gpt-oss-20b
export TT_CACHE_PATH=openai/gpt-oss-20b
```
```
pytest models/demos/gpt_oss/demo/text_demo.py -k "performance" --timeout=600
```

---

## Blackhole — bh_loudbox (8xP150)

Inherits all bh_llmbox rows (above), plus:

### llama3.1-8b — DP=8 performance
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and ci-32" --data_parallel 8 --max_seq_len 131072 --timeout=600
```

### llama3.1-8b — DP=8 stress
```
pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and stress" --data_parallel 8 --max_generated_tokens 22000 --timeout=3600
```

### gpt-oss-120b — performance
**Env**
```
export HF_MODEL=openai/gpt-oss-120b
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache
```
```
pytest models/demos/gpt_oss/demo/text_demo.py -k "performance" --timeout=1200
```

### flux.1-dev — LoudBox performance (bh_2x4sp0tp1)
**Env**: `HF_HUB_CACHE=/localdev/blackhole_demos/huggingface_data/black-forest-labs`
```
pytest models/tt_dit/tests/models/flux1/test_performance_flux1.py -k "bh_2x4sp0tp1" --timeout=600
```

### wan2.2-t2v-a14b — LoudBox performance
**Env**
```
export HF_HUB_CACHE=/localdev/blackhole_demos/huggingface_data/Wan-AI
export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
```
```
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "bh_2x4_sp1tp0 and resolution_480p and t2v" --timeout=1200
```

### mochi — LoudBox pipeline performance
```
TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE pytest models/tt_dit/tests/models/mochi/test_pipeline_mochi.py -k "4x8sp1tp0" --timeout=1500
```

---

## Blackhole — bh_quietbox_2 (2xP300)

Inherits all bh_llmbox rows (above) **except gpt-oss-120b**, plus:

### flux.1-dev — QuietBox 2 performance (2x2sp0tp1)
**Env**: `HF_HUB_CACHE=/localdev/blackhole_demos/huggingface_data/black-forest-labs`
```
pytest models/tt_dit/tests/models/flux1/test_performance_flux1.py -k "2x2sp0tp1" --timeout=600
```

### mochi — QuietBox 2 performance (2x2sp0tp1)
**Env**
```
export HF_HUB_CACHE=/localdev/blackhole_demos/huggingface_data/genmo
export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
```
```
pytest models/tt_dit/tests/models/mochi/test_performance_mochi.py -k "dit_2x2sp0tp1_vae_1x4sp0tp1_BH_QB" --timeout=1800
```

### wan2.2-t2v-a14b — QuietBox 2 performance
**Env**
```
export HF_HUB_CACHE=/localdev/blackhole_demos/huggingface_data/Wan-AI
export TT_DIT_CACHE_DIR=/tmp/TT_DIT_CACHE
```
```
pytest models/tt_dit/tests/models/wan2_2/test_performance_wan.py -k "2x2_sp0tp1 and resolution_480p and t2v" --timeout=1200
```

---

## Excluded (currently disabled in test defs)

| Model | SKU(s) | Reason |
|---|---|---|
| llama3.3-70b unit | wh_galaxy_perf | issue #42139 |
| gpt-oss-20b unit (T3K) | wh_llmbox_perf | segfault on CI |
| phi-4-mini unit/e2e | wh_n150, wh_n300 | shape assert |
| llama3.1-8b-dp e2e | wh_galaxy_perf | issue #42553 |
| llama3.3-70b-dp e2e | wh_galaxy_perf | issue #42553 |
| qwen2.5-72b e2e | wh_llmbox_perf | ci-token-matching hangs |
| mamba-2.8b unit/e2e | wh_n150, wh_n300 | PCC failures (issue #42163) |
| Several wh_n300 entries | gemma-4-e4b unit/e2e, shallow-unet unit, whisper unit/e2e | reduced n300 runner availability |

---

## Tier reference (which schedule a model lands in)

- **Tier 1** (`models-t1-*`): smallest, fastest models — gates fast feedback. Examples: llama3.1-8b (n150), whisper (n150), galaxy entries.
- **Tier 2** (`models-t2-*`): main coverage. Most mid-size LLMs and vision models.
- **Tier 3** (`models-t3-*`): long-running / lower-priority models (DP variants, 70B+ stress, full-vision).

The tier is per-(model, SKU) — the same model may be tier 1 on one SKU and tier 2 on another (e.g. llama3.1-8b is tier 1 on wh_n150 but tier 2 on wh_llmbox_perf).
