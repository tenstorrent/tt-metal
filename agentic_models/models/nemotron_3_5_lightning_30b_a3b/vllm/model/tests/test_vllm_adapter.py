# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the vLLM adapter contract: capabilities, call shapes and the vendored helpers. Opens no device."""

import importlib
import inspect
from types import SimpleNamespace

import pytest

PKG = 'agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt'
adapter_module = importlib.import_module(f'{PKG}.generator_vllm')
generator_module = importlib.import_module(f'{PKG}.generator')
tt_ccl_module = importlib.import_module(f'{PKG}.tt_ccl')
NemotronForCausalLM = adapter_module.NemotronForCausalLM

#: Keyword arguments vllm-tt-plugin f6fe76b passes (model_runner.submit_prefill, async_decode decode submit with
#: decode_input_update_contract >= 1, and the warmup calls) for a model with this adapter's capabilities.
PREFILL_KWARGS = {'tokens', 'page_table', 'kv_cache', 'enable_trace', 'prompt_lens', 'start_pos',
                  'page_tables_per_layer', 'sampling_params', 'empty_slots'}
DECODE_KWARGS = {'tokens', 'page_table', 'kv_cache', 'start_pos', 'page_tables_per_layer', 'sampling_params',
                 'slot_remap', 'reload_inputs', 'reload_page_table', 'reload_sampling_params',
                 'reset_sampling_state', 'enable_trace', 'read_from_device'}
WARMUP_PREFILL_KWARGS = {'kv_cache', 'can_sample_on_device', 'enable_trace'}
WARMUP_DECODE_KWARGS = {'kv_cache', 'max_batch_size', 'num_blocks', 'can_sample_on_device', 'enable_trace'}


def reloaded(monkeypatch, **env):
    for key in ('NEMOTRON_PREFIX_CACHE', 'NEMOTRON_UNBOUNDED_TOP_K'):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(adapter_module).NemotronForCausalLM


@pytest.fixture(autouse=True)
def restore_module():
    yield
    importlib.reload(adapter_module)


def test_capabilities(monkeypatch):
    caps = reloaded(monkeypatch).model_capabilities
    assert caps == {
        'supports_chunked_prefill': True,
        'supports_prefill_no_output': True,  # inert at plugin f6fe76b (8ea73b1 is not carried)
        'preserve_prefill_state_slots': True,  # plugin patch P1
        'supports_prefix_caching': False,
        'prefix_cache_recurrent_state_in_model': True,  # plugin patch P4, effective with --enable-prefix-caching
        'supports_sample_on_device': True,
        'supports_async_decode': True,
        'supports_device_penalties': False,
        'max_device_top_k': None,
    }
    # Contracts this model does not take: P2/P3 kwargs are never sent to it.
    assert not {'supports_prefill_sampling_mask', 'supports_original_prompt_lens'} & set(caps)
    assert reloaded(monkeypatch).decode_input_update_contract == 1


def test_capability_env(monkeypatch):
    assert reloaded(monkeypatch, NEMOTRON_PREFIX_CACHE='1').model_capabilities['supports_prefix_caching'] is True
    assert reloaded(monkeypatch, NEMOTRON_UNBOUNDED_TOP_K='0').model_capabilities['max_device_top_k'] == 64


@pytest.mark.parametrize(
    'method, kwargs',
    [
        ('prefill_forward', PREFILL_KWARGS),
        ('decode_forward', DECODE_KWARGS),
        ('warmup_model_prefill', WARMUP_PREFILL_KWARGS),
        ('warmup_model_decode', WARMUP_DECODE_KWARGS),
    ],
)
def test_plugin_call_shapes_bind(method, kwargs):
    signature = inspect.signature(getattr(NemotronForCausalLM, method))
    signature.bind(object(), **dict.fromkeys(kwargs))


def test_plugin_hooks_present():
    for name in ('initialize_vllm_model', 'get_max_tokens_all_users', 'get_kv_cache_spec',
                 'allocate_kv_cache_per_layer', 'read_decode_output', 'process_decode_output_host',
                 'release_request', 'note_state_slots_moved', 'release_persistent_capture'):
        assert callable(getattr(NemotronForCausalLM, name)), name


def mesh(shape=(1, 4)):
    return SimpleNamespace(shape=shape)


def test_initialize_geometry_and_precision(monkeypatch):
    for key in ('NEMOTRON_VLLM_HOST_COMPAT', 'NEMOTRON_VLLM_SERIAL_PREFILL', 'NEMOTRON_DECODE_BUCKETS',
                'NEMOTRON_VLLM_LAYERS', 'NEMOTRON_VLLM_EVENTS', 'NEMOTRON_PRECISION_CONFIG'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match='1x4'):
        NemotronForCausalLM.initialize_vllm_model(None, mesh((1, 8)), 32)
    with pytest.raises(ValueError, match='geometry'):
        NemotronForCausalLM.initialize_vllm_model(None, mesh(), 33)
    obj = NemotronForCausalLM.initialize_vllm_model(None, mesh(), 32, max_seq_len=262144)
    # The selected precision is read from doc/ next to tt/, the vendored layout.
    assert obj.precision['config_id'] == 'bfp8_hifi2_ccl8'
    assert set(obj.precision['weight_groups'].values()) == {'bfloat8_b'}
    assert obj.tt_supported_decode_batch_sizes == [1, 2, 4, 8, 16, 32]
    assert obj.batched_prefill is False and obj.host_compat is False
    with pytest.raises(ValueError, match='NEMOTRON_VLLM_HOST_COMPAT'):
        obj._mode('decode', None)
    monkeypatch.setenv('NEMOTRON_VLLM_HOST_COMPAT', '1')
    obj = NemotronForCausalLM.initialize_vllm_model(None, mesh(), 32)
    assert obj._mode('prefill', None) is False and obj._mode('decode', object()) is True


def test_kv_cache_spec():
    pytest.importorskip('vllm')
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    hf = SimpleNamespace(num_key_value_heads=2, head_dim=128, layers_block_type=['mamba', 'attention', 'moe'] * 2)
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf), cache_config=SimpleNamespace(block_size=32))
    spec = NemotronForCausalLM.get_kv_cache_spec(config)
    assert list(spec) == ['model.layers.0.self_attn', 'model.layers.1.self_attn']
    assert all(isinstance(s, FullAttentionSpec) and s.block_size == 32 for s in spec.values())
    config.cache_config.block_size = 16
    with pytest.raises(ValueError, match='block-size 32'):
        NemotronForCausalLM.get_kv_cache_spec(config)


def test_layer_kinds_accept_transformers_5_16_names():
    """transformers 5.16 reports mamba/attention layers as linear_attention/full_attention."""
    pytest.importorskip('vllm')
    from agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.tt.optimized_decoder import layer_kinds

    hf = SimpleNamespace(num_key_value_heads=2, head_dim=128,
                         layers_block_type=['linear_attention', 'full_attention', 'moe'] * 2)
    assert layer_kinds(hf) == ['mamba', 'attention', 'moe'] * 2
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=hf), cache_config=SimpleNamespace(block_size=32))
    assert list(NemotronForCausalLM.get_kv_cache_spec(config)) == ['model.layers.0.self_attn', 'model.layers.1.self_attn']


def test_initialize_restores_checkpoint_layer_names(monkeypatch):
    """vLLM's get_num_layers_by_block_type(..., 'attention') must count the attention layers under transformers 5.16."""
    hf = SimpleNamespace(layers_block_type=['linear_attention', 'full_attention', 'moe'] * 2)
    NemotronForCausalLM.initialize_vllm_model(hf, mesh(), 32)
    assert hf.layers_block_type == ['mamba', 'attention', 'moe'] * 2


def test_generator_uses_vendored_modules():
    readiness = importlib.import_module(f'{PKG}.readiness_contract')
    assert issubclass(generator_module.NemotronGenerator, readiness.Generator)
    assert generator_module.TT_CCL is tt_ccl_module.TT_CCL


class FakeTTNN:
    def __init__(self):
        self.created = 0

    def create_global_semaphore(self, mesh_device, cores, value):
        self.created += 1
        return f'sem{self.created}'

    @staticmethod
    def CoreRangeSet(ranges):
        return ranges

    @staticmethod
    def CoreRange(a, b):
        return (a, b)

    @staticmethod
    def CoreCoord(x, y):
        return (x, y)


@pytest.fixture
def fake_ttnn(monkeypatch):
    fake = FakeTTNN()
    pinned = importlib.import_module('models.common.tt_ccl')
    monkeypatch.setattr(tt_ccl_module, 'ttnn', fake)
    monkeypatch.setattr(pinned, 'ttnn', fake)
    return fake


def ccl_mesh():
    return SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))


def test_tt_ccl_lazy(fake_ttnn):
    ccl = tt_ccl_module.TT_CCL(ccl_mesh(), eager=False)
    assert fake_ttnn.created == 0
    assert ccl.sub_device_crs == {((0, 0), (10, 9))}
    first = ccl.get_and_cycle_barrier_semaphore_handle()
    assert fake_ttnn.created == 2
    ag = ccl.get_and_cycle_ag_semaphore_handles()
    assert fake_ttnn.created == 6 and len(ag) == 2
    # Later calls only cycle the double buffer.
    second = ccl.get_and_cycle_barrier_semaphore_handle()
    assert ccl.get_and_cycle_barrier_semaphore_handle() == first != second
    assert ccl.get_and_cycle_ag_semaphore_handles() != ag
    assert fake_ttnn.created == 6
    assert len(ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1)) == 3 and fake_ttnn.created == 12


def test_tt_ccl_eager_is_the_pinned_class(fake_ttnn):
    ccl = tt_ccl_module.TT_CCL(ccl_mesh())
    assert fake_ttnn.created == 3 * 2 * (1 + 2 + 3)
    assert ccl.get_and_cycle_barrier_semaphore_handle() == 'sem25'  # axis None: the third family
    assert fake_ttnn.created == 36
