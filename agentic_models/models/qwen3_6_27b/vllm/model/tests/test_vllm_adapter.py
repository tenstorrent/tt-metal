# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the vLLM adapter's contract with vllm-tt-plugin f6fe76b + the patches.json plugin patches.

No device is opened: the capabilities, the prefill kwargs the adapter hands the generator, the device top-k
width, the fused-MMRS NoC routing and the vendored kda kernel text are checked on the host.
"""

import ast
import importlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PKG = 'agentic_models.models.qwen3_6_27b.vllm'
TT = Path(__file__).resolve().parents[2] / 'tt'


def capabilities(**env):
    """model_capabilities as the class body computes them under `env` (a fresh process: the body reads env once)."""
    script = f'import json; from {PKG}.tt.generator_vllm import QwenVLLM as Q; print(json.dumps(Q.model_capabilities))'
    full = {k: v for k, v in os.environ.items() if not k.startswith('QWEN_VLLM_')}
    full.update(env)
    out = subprocess.run([sys.executable, '-c', script], env=full, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_capabilities_default():
    caps = capabilities()
    assert caps == {
        'supports_chunked_prefill': True,
        'supports_prefix_caching': False,
        'supports_sample_on_device': True,
        'supports_prefill_sampling_mask': True,
        'supports_original_prompt_lens': True,
        'preserve_prefill_state_slots': True,
        'prefix_cache_recurrent_state_in_model': False,
        'max_device_top_k': 32,
        'supports_device_penalties': True,
        'supports_device_grammar': False,
        'supports_async_decode': True,
    }
    # The bring-up plugin's name for original_prompt_lens is gone.
    assert 'supports_prefill_sampling_history' not in caps


def test_capabilities_serving_spec():
    """The TTI spec's env: APC with 32 recurrent-state snapshots and MTP K=3."""
    caps = capabilities(QWEN_VLLM_PREFIX_CACHE='32', QWEN_VLLM_SPEC_K='3')
    assert caps['supports_prefix_caching'] is True
    assert caps['prefix_cache_recurrent_state_in_model'] is True
    assert caps['supports_spec_decode'] is True
    assert caps['spec_requirements'] == ['device_propose']
    assert caps['spec_ids_are_sampled'] is True and caps['supports_async_spec_decode'] is True
    # The plugin validates these as booleans at platform init.
    for key in ('supports_prefill_sampling_mask', 'supports_original_prompt_lens', 'preserve_prefill_state_slots',
                'prefix_cache_recurrent_state_in_model'):
        assert type(caps[key]) is bool, key


def test_spec_plan_declines_drafts_above_16_live_rows():
    """No [B>16, W>=2] verify shape exists, so k_by_rows caps 17..max_num_seqs rows at 0 drafts (needs the plugin
    patch allowing k = 0 for supports_narrow_decode models; f6fe76b keeps proposals across a peer's prefill)."""
    script = (
        'import json, sys, types\n'
        'm = types.ModuleType("vllm_tt_plugin.spec_decode")\n'
        'm.SpecPlan = lambda **kw: kw\n'
        'm.SpecReject = lambda **kw: kw\n'
        'sys.modules["vllm_tt_plugin.spec_decode"] = m\n'
        f'from {PKG}.tt.generator_vllm import QwenVLLM as Q\n'
        'p = Q.spec_plan(None, 32, 3)\n'
        'print(json.dumps({"k": p["effective_k"], "rows": p["k_by_rows"], "narrow": p["supports_narrow_decode"]}))'
    )
    full = {k: v for k, v in os.environ.items() if not k.startswith('QWEN_VLLM_')}
    full['QWEN_VLLM_SPEC_K'] = '3'
    out = subprocess.run([sys.executable, '-c', script], env=full, capture_output=True, text=True, check=True)
    plan = json.loads(out.stdout.strip().splitlines()[-1])
    assert plan == {'k': 3, 'rows': [[8, 3], [16, 1], [32, 0]], 'narrow': True}


class FakeGenerator:
    def __init__(self):
        self.cache = object()
        self.page_host = torch.zeros((32, 16), dtype=torch.int32)
        self.traces = {}
        self.counters = {}
        self.calls = []

    def prefill_forward(self, tokens, **kwargs):
        self.calls.append(kwargs)
        return torch.zeros((len(tokens), 1), dtype=torch.int64)


def adapter():
    from agentic_models.models.qwen3_6_27b.vllm.tt.generator_vllm import QwenVLLM

    a = QwenVLLM.__new__(QwenVLLM)
    a.generator = FakeGenerator()
    a.model = SimpleNamespace(config=SimpleNamespace(vocab_size=248320))
    a.mesh = SimpleNamespace(num_program_cache_entries=lambda: 0)
    a.host_compatibility = True
    a.request_specific_rope = False
    a.active_decode_rows = 0
    return a


def prefill(a, sampling_params, **kwargs):
    a.prefill_forward(
        torch.zeros((2, 64), dtype=torch.int32),
        page_table=torch.zeros((2, 2), dtype=torch.int32),
        kv_cache=a.generator.cache,
        sampling_params=sampling_params,
        prompt_lens=[64, 40],
        start_pos=[32, 0],
        empty_slots=[3, 5],
        **kwargs,
    )
    return a.generator.calls[-1]


def test_prefill_mask_reaches_the_generator_on_the_device_route():
    a = adapter()
    call = prefill(a, object(), prefill_sampling_mask=[False, True], original_prompt_lens=[100, 40])
    assert call['sampling_mask'] == [False, True]
    assert 'prefill_sampling_mask' not in call
    assert call['original_prompt_lens'] == [100, 40]
    assert call['empty_slots'] == [3, 5] and call['start_pos'] == [32, 0] and call['prompt_lens'] == [64, 40]


def test_prefill_mask_is_dropped_on_the_host_route():
    """The plugin forwards the mask on host routes too; the generator only accepts it with device sampling."""
    a = adapter()
    call = prefill(a, None, prefill_sampling_mask=[False, True], original_prompt_lens=[100, 40])
    assert 'sampling_mask' not in call and 'prefill_sampling_mask' not in call
    assert call['original_prompt_lens'] == [100, 40]


def test_prefill_without_the_plugin_patches():
    """Plain f6fe76b passes neither kwarg: every row is final."""
    call = prefill(adapter(), object())
    assert 'sampling_mask' not in call and 'original_prompt_lens' not in call


def test_generator_takes_the_kwargs_the_adapter_forwards():
    from agentic_models.models.qwen3_6_27b.vllm.tt.generator import QwenGenerator

    params = inspect.signature(QwenGenerator.prefill_forward).parameters
    for name in ('sampling_mask', 'original_prompt_lens', 'start_pos', 'prompt_lens', 'empty_slots'):
        assert name in params, name


def keyword_values(path, func_suffix, keyword):
    tree = ast.parse(path.read_text())
    return [
        k.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func).endswith(func_suffix)
        for k in n.keywords
        if k.arg == keyword
    ]


def test_device_sampler_width_is_32():
    """ttnn.sampling at the tt-metal pin takes k <= 32; both generator samplers are built for 32."""
    tree = ast.parse((TT / 'generator.py').read_text())
    widths = [
        ast.literal_eval(k.value) for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords
        if k.arg == 'max_top_k'
    ]
    assert widths == [32, 32]
    from agentic_models.models.qwen3_6_27b.vllm.tt.sampling.generator import SamplingParams, format_sampling_params

    formatted = format_sampling_params(
        SamplingParams(temperature=[1.0, 1.0, 1.0], top_k=[20, 40, 0], top_p=[0.95, 0.95, 0.95]), 32, max_top_k=32
    )
    assert formatted.top_k[:3] == [20, 32, 32]


def test_vendored_sampling_is_used():
    gen = importlib.import_module(f'{PKG}.tt.generator')
    sampling = importlib.import_module(f'{PKG}.tt.sampling.generator')
    assert gen.SamplingGenerator is sampling.SamplingGenerator
    assert sampling.TTSampling.__module__ == f'{PKG}.tt.sampling.tt_sampling'


def test_fused_mmrs_uses_the_dedicated_read_noc():
    """tt_metal patch qwen3_6_27b-matmul-reduce-scatter-prefill-dedicated-read-noc: the validated NoC routing."""
    values = keyword_values(TT / 'multichip_decoder.py', 'matmul_reduce_scatter_prefill', 'dedicated_read_noc')
    assert [ast.literal_eval(v) for v in values] == [True]


def test_kda_kernel_text_is_vendored():
    module = importlib.import_module(f'{PKG}.tt.tiled_prefill_conv_candidate')
    assert module.NATIVE_COMPUTE.parent == TT / 'kernels'
    module.native_compute.cache_clear()
    source = module.native_compute()  # asserts the one template signature it rewrites is present
    assert 'void kernel_main()' in source and 'dfb::' not in source


def test_beta_decay_kernel_uses_the_pin_softplus_init():
    """At the tt-metal pin softplus_tile_init() takes no template argument (base 8c48b4e4's <true> fails to JIT)."""
    from agentic_models.models.qwen3_6_27b.vllm.tt.optimized_decoder import _gate178_COMPUTE

    assert 'softplus_tile_init();' in _gate178_COMPUTE and 'softplus_tile_init<' not in _gate178_COMPUTE


def test_default_precision_is_config_p():
    from agentic_models.models.qwen3_6_27b.vllm.tt.precision import load_precision

    assert load_precision()['config_id'] == 'bfp8_baseline'


@pytest.mark.parametrize('module', sorted(p.stem for p in TT.glob('*.py') if p.stem != '__init__'))
def test_module_imports_without_a_device(module):
    importlib.import_module(f'{PKG}.tt.{module}')
