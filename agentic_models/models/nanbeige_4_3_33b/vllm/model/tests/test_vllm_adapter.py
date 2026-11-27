# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the vLLM adapter's contract with vllm-tt-plugin and the pinned tt-metal. No device is opened.

Covers the capabilities the plugin reads (and the plugin patches they enable), the prefill kwargs the adapter
translates, the opt-in tt-metal switches it sets, the device top-k width and the KV pool, and that register()
replaces the plugin's builtin registration of the architecture.
"""

import inspect
import os

import pytest
import torch

from agentic_models.models.nanbeige_4_3_33b.vllm.model import entry
from agentic_models.models.nanbeige_4_3_33b.vllm.tt import decoder
from agentic_models.models.nanbeige_4_3_33b.vllm.tt import generator as canonical
from agentic_models.models.nanbeige_4_3_33b.vllm.tt.generator_vllm import NanbeigeVLLM
from agentic_models.models.nanbeige_4_3_33b.vllm.tt.precision import load_policy


def test_capabilities():
    caps = NanbeigeVLLM.model_capabilities
    # Shared vllm-tt-plugin patches P2 / P3 and the package's own plugin patch.
    assert caps["supports_prefill_sampling_mask"] is True
    assert caps["supports_original_prompt_lens"] is True
    assert caps["supports_device_top_k_unset"] is True
    assert caps["reconcile_async_grammar_rows"] is True
    assert caps["supports_chunked_prefill"] is True and caps["supports_prefix_caching"] is True
    assert caps["supports_sample_on_device"] is True and caps["supports_device_penalties"] is True
    assert NanbeigeVLLM.decode_input_update_contract == 1


def test_device_top_k_is_64_with_the_opt_in_sampler():
    # top_k <= 64 and unset top-k sample on device; the 64-candidate ttnn.sampling program is the
    # nanbeige_4_3_33b-sampling-top64 tt_metal patch, which the adapter switches on at import.
    assert NanbeigeVLLM.model_capabilities["max_device_top_k"] == 64
    assert os.environ["TTNN_NANBEIGE_SAMPLING_TOP64"] == "1"
    assert os.environ.get("TTNN_NANBEIGE_PRECISION") in ("0", "1")


def test_fused_residual_all_reduce_is_off_at_the_pin():
    # all_reduce_async(residual_tensor=..., one_shot=...) is newer than the pinned tt-metal.
    assert decoder.FUSED_RESIDUAL_ALL_REDUCE is False


class Recorder:
    def __init__(self):
        self.calls = []

    def prefill_forward(self, *args, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros(len(kwargs["prompt_lens"]), 1, dtype=torch.int64)


def adapter_with(recorder, vocab_size=166912):
    model = NanbeigeVLLM()
    model.generator = recorder
    model.config = type("Config", (), {"vocab_size": vocab_size})()
    model.mesh = type("Mesh", (), {"num_program_cache_entries": lambda self: 0})()
    return model


def test_prefill_sampling_mask_reaches_the_generator_as_sampling_mask():
    recorder = Recorder()
    adapter_with(recorder).prefill_forward(
        torch.zeros(2, 4, dtype=torch.int64), page_table=None, prompt_lens=[4, 4], start_pos=[0, 0],
        prefill_sampling_mask=[True, False],
    )
    (kwargs,) = recorder.calls
    assert "prefill_sampling_mask" not in kwargs
    assert kwargs["sampling_mask"].dtype == torch.bool and kwargs["sampling_mask"].tolist() == [True, False]


def test_generator_prefill_accepts_the_plugin_kwargs():
    params = inspect.signature(canonical.Generator.prefill_forward).parameters
    assert {"sampling_mask", "original_prompt_lens", "start_pos", "sampling_params"} <= set(params)


def test_kv_pool_fits_the_policy():
    policy = load_policy()
    assert policy["kv_dtype"] == "bfp8" and policy["kv_pool_tokens"] == 589824
    assert NanbeigeVLLM.get_max_tokens_all_users(max_num_seqs=16) == 589824 - 32 * 16


def test_register_replaces_the_builtin_llama():
    from vllm.model_executor.models.registry import ModelRegistry

    ModelRegistry.register_model("TTLlamaForCausalLM", "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM")
    assert entry.register() == "TTLlamaForCausalLM"
    assert ModelRegistry.models["TTLlamaForCausalLM"].class_name == "NanbeigeVLLM"
