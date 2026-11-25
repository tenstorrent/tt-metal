# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the vLLM adapter's contract with vllm-tt-plugin. No device is opened.

Covers the capabilities the plugin reads (including the two plugin patches the model needs), the device top-k
width (32, the ttnn.sampling limit at the pinned tt-metal), the vendored sampling modules, the prefill kwargs, the
KV-cache spec and pool size, and that register() replaces the plugin's builtin registration of the architecture.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from vllm_tt_plugin.model_input import TTSamplingParams

import ttnn
from agentic_models.models.ornith_1_5_9b.vllm.model import entry
from agentic_models.models.ornith_1_5_9b.vllm.tt import cache_contract
from agentic_models.models.ornith_1_5_9b.vllm.tt import generator as canonical
from agentic_models.models.ornith_1_5_9b.vllm.tt import prefix_snapshots as PS
from agentic_models.models.ornith_1_5_9b.vllm.tt.generator_vllm import OrnithForCausalLM, OrnithMTPForCausalLM
from agentic_models.models.ornith_1_5_9b.vllm.tt.sampling import generator as sampling
from agentic_models.models.ornith_1_5_9b.vllm.tt.sampling import tt_sampling

ADAPTERS = [OrnithForCausalLM, OrnithMTPForCausalLM]


@pytest.mark.parametrize("cls", ADAPTERS)
def test_capabilities(cls):
    caps = cls.model_capabilities
    # vllm-tt-plugin patches: P1 keeps a continuing chunked prefill in its state slot, P3 passes original_prompt_lens.
    assert caps["preserve_prefill_state_slots"] is True
    assert caps["supports_original_prompt_lens"] is True
    assert caps["supports_chunked_prefill"] is True
    assert caps["supports_sample_on_device"] is True and caps["supports_device_penalties"] is True
    assert caps["supports_prefix_caching"] == PS.ENABLED
    assert caps["fabric_config"] == {"config": ttnn.FabricConfig.FABRIC_1D}
    assert cls.decode_input_update_contract == 1


@pytest.mark.parametrize("cls", ADAPTERS)
def test_device_top_k_is_32(cls):
    # The plugin samples a request on device only when its top_k <= max_device_top_k; above 32 goes to the host.
    assert cls.model_capabilities["max_device_top_k"] == canonical.DEVICE_SAMPLING_MAX_TOP_K == 32


def test_generator_uses_the_vendored_sampling_modules():
    assert canonical.SamplingGenerator is sampling.SamplingGenerator
    assert canonical.format_sampling_params is sampling.format_sampling_params
    assert sampling.TTSampling is tt_sampling.TTSampling


def stochastic(top_k, temperature=1.0):
    n = len(top_k)
    return sampling.SamplingParams(temperature=[temperature] * n, top_k=list(top_k), top_p=[0.95] * n)


@pytest.mark.parametrize("k, device_k", [(1, 1), (20, 20), (32, 32), (40, 32), (0, 32), (-1, 32)])
def test_format_caps_top_k_at_32(k, device_k):
    formatted = sampling.format_sampling_params(stochastic([k]), 32, max_top_k=canonical.DEVICE_SAMPLING_MAX_TOP_K)
    assert formatted.top_k[0] == device_k
    assert formatted.top_k[1:] == [1] * 31


@pytest.mark.parametrize("width", [16, 33, 50, 128, "32"])
def test_sampler_width_must_be_32_or_64(width):
    with pytest.raises(ValueError, match="32 or 64"):  # allow-pytest.raises: no tt-metal conftest
        sampling.format_sampling_params(stochastic([20]), 32, max_top_k=width)
    # TTSampling validates the width before it touches the mesh device.
    with pytest.raises(ValueError, match="32 or 64"):  # allow-pytest.raises: no tt-metal conftest
        tt_sampling.TTSampling(mesh_device=None, tt_ccl=None, args=SimpleNamespace(max_top_k=width))


@pytest.mark.parametrize("k", [1, 20, 32])
def test_device_validation_accepts_top_k_up_to_32(k):
    formatted = canonical.OrnithGenerator._validate_sampling(None, stochastic([k]))
    assert formatted.top_k[0] == k


@pytest.mark.parametrize("k", [33, 40, 64, 0])
def test_device_validation_rejects_stochastic_top_k_above_32(k):
    with pytest.raises(ValueError, match=r"top_k in \[1,32\]"):  # allow-pytest.raises: no tt-metal conftest
        canonical.OrnithGenerator._validate_sampling(None, stochastic([k]))


def test_greedy_rows_accept_any_top_k():
    params = sampling.SamplingParams(temperature=[0.0, 0.0], top_k=[64, 0], top_p=[1.0, 1.0])
    formatted = canonical.OrnithGenerator._validate_sampling(None, params)
    assert formatted.top_k[:2] == [1, 1]


def test_plugin_sampling_params_convert_to_the_vendored_dataclass():
    params = TTSamplingParams(
        temperature=[0.6, 0.0], top_k=[20, 1], top_p=[0.95, 1.0], presence_penalty=[1.5, 0.0], seed=[7, 8]
    )
    converted = OrnithForCausalLM._sampling_params(params)
    assert isinstance(converted, sampling.SamplingParams)
    assert converted.top_k == [20, 1] and converted.presence_penalty == [1.5, 0.0] and converted.seed == [7, 8]
    assert OrnithForCausalLM._sampling_params(None) is None


def test_prefill_kwargs():
    names = inspect.signature(OrnithForCausalLM.prefill_forward).parameters
    for name in ("prompt_lens", "start_pos", "empty_slots", "enable_trace", "sampling_params", "original_prompt_lens"):
        assert name in names, name
    decode = inspect.signature(OrnithForCausalLM.decode_forward).parameters
    assert decode["kwargs"].kind is inspect.Parameter.VAR_KEYWORD  # reload_*, read_from_device, ... pass through


def test_prefill_requires_original_prompt_lens():
    # Without plugin patch P3 the plugin sends no original_prompt_lens; the adapter refuses rather than guessing.
    adapter = object.__new__(OrnithForCausalLM)
    adapter._check_cache = lambda cache: None
    with pytest.raises(ValueError, match="original_prompt_lens"):  # allow-pytest.raises: no tt-metal conftest
        adapter.prefill_forward(torch.zeros(1, 32, dtype=torch.int64), None, None, [32], empty_slots=[0])
    adapter.max_seq_len = 262144
    with pytest.raises(TypeError, match="original_prompt_lens"):  # allow-pytest.raises: no tt-metal conftest
        adapter.prefill_forward(None, None, None, [32], empty_slots=[0], original_prompt_lens=torch.tensor([32]))


def test_kv_cache_spec_and_pool():
    config = SimpleNamespace(cache_config=SimpleNamespace(block_size=32, mamba_cache_mode="align"))
    spec = OrnithForCausalLM.get_kv_cache_spec(config)
    assert sorted(spec) == sorted(f"model.layers.{i}.self_attn" for i in cache_contract.FULL_LAYERS)
    assert len(spec) == 8 and all(s.block_size == 32 for s in spec.values())
    # Prefix caching keeps GDN state in model snapshots, so vLLM's hybrid align mode is turned off.
    assert config.cache_config.mamba_cache_mode == ("none" if PS.ENABLED else "align")
    with pytest.raises(ValueError, match="32 tokens"):  # allow-pytest.raises: no tt-metal conftest
        OrnithForCausalLM.get_kv_cache_spec(SimpleNamespace(cache_config=SimpleNamespace(block_size=64)))
    # Default shared pool from the selected BFP8 KV policy (doc/datatype_sweep/selected_precision_config.json).
    assert OrnithForCausalLM.get_max_tokens_all_users("ornith", 4, 1, 262144, 32) == 1048576


def test_selected_precision_is_shipped():
    from agentic_models.models.ornith_1_5_9b.vllm.tt.precision import DEFAULT_PATH, load_policy

    assert DEFAULT_PATH.exists()
    assert load_policy()["id"] == "bfp8_all_hifi2"


def test_mtp_spec_plan_limits():
    from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

    assert isinstance(OrnithMTPForCausalLM.spec_plan(None, 16, 3), SpecPlan)
    assert isinstance(OrnithMTPForCausalLM.spec_plan(None, 32, 3), SpecReject)


def test_register_replaces_the_builtin_registration(monkeypatch):
    from vllm.model_executor.models.registry import ModelRegistry

    monkeypatch.setattr(ModelRegistry, "models", dict(ModelRegistry.models))
    ModelRegistry.register_model(entry.ARCH, "vllm_tt_plugin.builtin_placeholder:TTQwen3_5")
    assert entry.register() == entry.ARCH
    registered = ModelRegistry.models[entry.ARCH]
    assert f"{registered.module_name}:{registered.class_name}" == entry.METADATA["main_class"]
