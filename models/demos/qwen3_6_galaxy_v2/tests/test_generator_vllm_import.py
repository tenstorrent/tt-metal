# SPDX-License-Identifier: Apache-2.0
"""Off-device structural checks for the qwen3.6 vLLM generator wrapper."""
import importlib
import inspect


def test_generator_vllm_uses_local_v2_modules():
    mod = importlib.import_module("models.demos.qwen3_6_galaxy_v2.tt.generator_vllm")
    src = inspect.getsource(mod)
    # Must NOT import the parent llama3_70b_galaxy model/config classes.
    assert "llama3_70b_galaxy.tt.llama_model" not in src
    assert "llama3_70b_galaxy.tt.model_config" not in src
    assert "llama3_70b_galaxy.tt.qwen_model_config" not in src
    # Must use the local v2 qwen36 args + local transformer.
    assert "qwen3_6_galaxy_v2.tt.qwen36_model_config" in src
    assert "qwen3_6_galaxy_v2.tt.llama_model" in src
    assert "qwen3_6_galaxy_v2.tt.generator" in src


def test_serving_class_exists_and_has_vllm_api():
    mod = importlib.import_module("models.demos.qwen3_6_galaxy_v2.tt.generator_vllm")
    cls = getattr(mod, "Qwen3_5ForConditionalGeneration")
    assert hasattr(cls, "initialize_vllm_model")
    assert hasattr(cls, "allocate_kv_cache")
    # Must NOT call AutoModelForCausalLM (broken for this checkpoint).
    assert "AutoModelForCausalLM" not in inspect.getsource(mod)


def test_no_prefetcher_perf_mode_kwarg():
    mod = importlib.import_module("models.demos.qwen3_6_galaxy_v2.tt.generator_vllm")
    src = inspect.getsource(mod)
    assert "enable_prefetcher_performance_mode=True" not in src
