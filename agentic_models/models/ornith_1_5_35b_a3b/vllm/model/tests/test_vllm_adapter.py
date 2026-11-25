# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the vLLM adapter contract. No device is opened.

Covers the capabilities the plugin reads (including the P1 / P2 plugin-patch capabilities), the device top-k
limit of 32, the keyword arguments the plugin passes to prefill_forward / decode_forward, the measured KV
budget gate, the forked-op call sites and the overwriting architecture registration.
"""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agentic_models.models.ornith_1_5_35b_a3b.vllm.model import entry
from agentic_models.models.ornith_1_5_35b_a3b.vllm.tt.generator import OrnithGenerator
from agentic_models.models.ornith_1_5_35b_a3b.vllm.tt.generator_vllm import OrnithForCausalLM
from agentic_models.models.ornith_1_5_35b_a3b.vllm.tt.model import measured_kv_capacity
from agentic_models.models.ornith_1_5_35b_a3b.vllm.tt.precision import load_precision

TT = Path(__file__).resolve().parents[2] / 'tt'


def test_capabilities():
    caps = OrnithForCausalLM.model_capabilities
    assert caps['preserve_prefill_state_slots'] is True  # plugin patch P1
    assert caps['supports_prefill_sampling_mask'] is True  # plugin patch P2
    assert caps['supports_chunked_prefill'] is True
    assert caps['supports_prefix_caching'] is False  # recurrent state: APC stays off
    assert caps['supports_sample_on_device'] and caps['supports_device_penalties'] and caps['supports_async_decode']
    assert caps['max_device_top_k'] == 32  # ttnn.sampling takes k <= 32; larger top_k samples on the host
    assert caps['output_tokens_per_step'] == 1 and not caps['supports_spec_decode']
    assert OrnithForCausalLM.decode_input_update_contract == 1
    # P1 is the adapter's declaration; the generator keeps the bring-up set.
    assert 'preserve_prefill_state_slots' not in OrnithGenerator.model_capabilities
    assert {k: v for k, v in caps.items() if k != 'preserve_prefill_state_slots'} == OrnithGenerator.model_capabilities


def generator(batch_size=2):
    g = object.__new__(OrnithGenerator)
    g.batch_size = batch_size
    return g


@pytest.mark.parametrize('k', [1, 20, 32])
def test_device_top_k_up_to_32(k):
    values = generator()._sampling_values(top_k=k, top_p=0.95, temperature=1.0, seed=[1, 2])
    assert values['top_k'] == [k, k] and values['seed'] == [1, 2]


@pytest.mark.parametrize('k', [33, 64, 0])
def test_device_top_k_outside_1_to_32_rejected(k):
    with pytest.raises(ValueError, match=r'top_k 1\.\.32'):  # allow-pytest.raises: no tt-metal conftest
        generator()._sampling_values(top_k=k, top_p=0.95, temperature=1.0, seed=1)


def test_greedy_rows_ignore_top_k():
    values = generator()._sampling_values(top_k=[64, 20], top_p=1.0, temperature=[0.0, 1.0], seed=1)
    assert values['top_k'] == [1, 20]


def test_plugin_keyword_arguments_are_accepted():
    prefill = inspect.signature(OrnithGenerator.prefill_forward).parameters
    for name in ('page_table', 'kv_cache', 'prompt_lens', 'start_pos', 'empty_slots', 'sampling_params',
                 'prefill_sampling_mask', 'page_tables_per_layer', 'enable_trace'):
        assert name in prefill, name
    decode = inspect.signature(OrnithGenerator.decode_forward).parameters
    for name in ('page_table', 'kv_cache', 'sampling_params', 'read_from_device', 'enable_trace', 'reload_inputs',
                 'reload_page_table', 'reload_sampling_params', 'reset_sampling_state', 'slot_remap',
                 'page_tables_per_layer', 'prompt_tokens', 'output_tokens'):
        assert name in decode, name


class RecordingGenerator:
    def __init__(self, batch_size):
        self.pages_host = torch.zeros(batch_size, 8, dtype=torch.int32)
        self.calls = []

    def _position(self, slot):
        return 0

    def prefill_forward(self, tokens, **kwargs):
        self.calls.append(kwargs)
        return 'out'


def adapter(batch_size=4):
    a = object.__new__(OrnithForCausalLM)
    a.batch_size, a.context, a.benchmark_timing = batch_size, 1024, None
    a.generator = RecordingGenerator(batch_size)
    return a


def test_prefill_forwards_mask_and_slots():
    a = adapter()
    mask, params, starts = [False, True], object(), torch.tensor([0, 64])
    out, deltas = a.prefill_forward(
        torch.zeros(2, 96, dtype=torch.long),
        page_table=torch.tensor([[3, 4], [5, 6]]),
        kv_cache='cache',
        prompt_lens=[64, 96],
        empty_slots=[2, 0],
        start_pos=starts,
        sampling_params=params,
        prefill_sampling_mask=mask,
        pixel_values=[None, None],  # vLLM's empty multimodal metadata for text requests
    )
    assert out == 'out' and torch.equal(deltas, torch.zeros(2, dtype=torch.long))
    (call,) = a.generator.calls
    assert 'pixel_values' not in call
    assert call['empty_slots'] == [2, 0] and call['prompt_lens'] == [64, 96] and call['kv_cache'] == 'cache'
    assert call['prefill_sampling_mask'] is mask and call['sampling_params'] is params and call['start_pos'] is starts
    assert call['page_table'][2, :2].tolist() == [3, 4] and call['page_table'][0, :2].tolist() == [5, 6]


def test_prefill_rejects_multimodal_inputs():
    with pytest.raises(ValueError, match='pixel_values'):  # allow-pytest.raises: no tt-metal conftest
        adapter().prefill_forward(
            torch.zeros(1, 8, dtype=torch.long),
            page_table=torch.tensor([[1]]),
            kv_cache='cache',
            prompt_lens=[8],
            pixel_values=[torch.zeros(1)],
        )


def test_measured_kv_budget():
    # The capacity gate hashes the vendored runtime sources and requires the selected BF16 KV policy.
    assert load_precision()['kv_cache_dtype'] == 'bfloat16'
    record = measured_kv_capacity()
    assert record['physical_num_blocks'] * 32 == record['physical_token_capacity']
    budget = OrnithGenerator.get_max_tokens_all_users('ornith', 4, 1, 262144, 32)
    assert budget == record['max_tokens_all_users']
    assert OrnithGenerator.get_max_tokens_all_users('ornith', 4, 1, 4096, 2) == 8192
    with pytest.raises(ValueError):  # allow-pytest.raises: no tt-metal conftest
        OrnithGenerator.get_max_tokens_all_users('ornith', 8, 1, 262144, 32)


def calls(path):
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            found.append(ast.unparse(node.func))
    return found


@pytest.mark.parametrize('name, expected', [('multichip_decoder.py', (2, 1)), ('optimized_decoder.py', (4, 2))])
def test_moe_ops_use_the_bringup_forks(name, expected):
    found = calls(TT / name)
    assert (found.count('ttnn.bringup.moe_fused_swiglu'), found.count('ttnn.bringup.post_combine_reduce')) == expected
    assert not [f for f in found if f.endswith(('deepseek_prefill.moe_fused_swiglu', 'deepseek_prefill.post_combine_reduce'))]


def test_register_overwrites_existing_architecture(monkeypatch):
    from vllm.model_executor.models.registry import ModelRegistry

    monkeypatch.setattr(ModelRegistry, 'models', dict(ModelRegistry.models))
    ModelRegistry.register_model(entry.ARCH, 'some.builtin.module:TTQwen3_5MoeDemo')
    assert entry.register() == entry.ARCH
    module, _, cls = entry.METADATA['main_class'].partition(':')
    registered = ModelRegistry.models[entry.ARCH]
    assert (registered.module_name, registered.class_name) == (module, cls)
    assert entry.register() == entry.ARCH  # idempotent
