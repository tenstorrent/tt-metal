# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from laguna_vllm_ext import hybrid_kv


def _layer_types():
    return ["full_attention" if layer % 4 == 0 else "sliding_attention" for layer in range(40)]


def _config():
    hf_config = SimpleNamespace(
        model_type="laguna",
        num_hidden_layers=40,
        layer_types=_layer_types(),
        sliding_window=512,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            max_model_len=131072,
        ),
        cache_config=SimpleNamespace(
            block_size=64,
            enable_prefix_caching=False,
        ),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=True,
            max_num_batched_tokens=8192,
            max_num_seqs=1,
        ),
    )


@pytest.fixture(autouse=True)
def _hybrid_enabled(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_HYBRID_KV", "1")


def test_exact_block_floor_covers_full_three_sliding_groups_and_null_block():
    config = _config()

    # full=ceil(131072/64)=2048; each sliding group holds
    # ceil((511+8192)/64)+1=137; block zero is the global null block.
    assert hybrid_kv.exact_hybrid_kv_num_blocks(config) == 2048 + 3 * 137 + 1
    assert hybrid_kv.exact_hybrid_kv_num_blocks(config) == 2460


def test_exact_block_floor_scales_full_group_for_262k_probe_only():
    config = _config()
    config.model_config.max_model_len = 262144

    # The sliding-group reservation remains bounded by the 8K scheduler step,
    # while the one full-attention group doubles from 2048 to 4096 blocks.
    assert hybrid_kv.exact_hybrid_kv_num_blocks(config) == 4096 + 3 * 137 + 1
    assert hybrid_kv.exact_hybrid_kv_num_blocks(config) == 4508


def test_worker_patch_raises_underallocated_plugin_heuristic_and_is_idempotent():
    worker = SimpleNamespace(
        get_num_available_blocks_tt=lambda _config, _num_devices=1: 2113,
    )
    assert hybrid_kv._patch_worker(worker)
    assert not hybrid_kv._patch_worker(worker)

    assert worker.get_num_available_blocks_tt(_config(), 2) == 2460


def test_worker_patch_preserves_larger_pool_and_is_inert_when_disabled(monkeypatch):
    worker = SimpleNamespace(
        get_num_available_blocks_tt=lambda _config, _num_devices=1: 3000,
    )
    hybrid_kv._patch_worker(worker)
    assert worker.get_num_available_blocks_tt(_config(), 2) == 3000

    monkeypatch.setenv("TT_LAGUNA_HYBRID_KV", "0")
    assert worker.get_num_available_blocks_tt(_config(), 2) == 3000


def _fake_platform(module):
    class FakeTTPlatform:
        calls = 0

        @classmethod
        def check_and_update_config(cls, config):
            cls.calls += 1
            if config.model_config.hf_config.model_type not in module._CHUNKED_PREFILL_MODEL_TYPES:
                config.scheduler_config.enable_chunked_prefill = False
                config.scheduler_config.max_num_batched_tokens = config.model_config.max_model_len

    return FakeTTPlatform


def test_platform_patch_temporarily_admits_only_enabled_laguna_chunking():
    module = SimpleNamespace(_CHUNKED_PREFILL_MODEL_TYPES={"gemma4"})
    platform = _fake_platform(module)
    assert hybrid_kv._patch_platform(module, platform)
    assert not hybrid_kv._patch_platform(module, platform)

    config = _config()
    platform.check_and_update_config(config)

    assert platform.calls == 1
    assert config.scheduler_config.enable_chunked_prefill is True
    assert config.scheduler_config.max_num_batched_tokens == 8192
    assert module._CHUNKED_PREFILL_MODEL_TYPES == {"gemma4"}


def test_platform_patch_retains_pinned_plugin_policy_when_hybrid_is_off(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_HYBRID_KV", "0")
    module = SimpleNamespace(_CHUNKED_PREFILL_MODEL_TYPES={"gemma4"})
    platform = _fake_platform(module)
    hybrid_kv._patch_platform(module, platform)
    config = _config()

    platform.check_and_update_config(config)

    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.max_num_batched_tokens == 131072


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: setattr(c.cache_config, "enable_prefix_caching", True), "prefix caching disabled"),
        (lambda c: setattr(c.cache_config, "block_size", 32), "block_size=64"),
        (lambda c: setattr(c.scheduler_config, "enable_chunked_prefill", False), "scheduler chunked prefill"),
        (lambda c: setattr(c.scheduler_config, "max_num_batched_tokens", 4096), "max_num_batched_tokens=8192"),
        (lambda c: setattr(c.scheduler_config, "max_num_seqs", 2), "max_num_seqs=1"),
        (lambda c: setattr(c.model_config.hf_config, "model_type", "other"), "Laguna-specific"),
        (lambda c: c.model_config.hf_config.layer_types.__setitem__(17, "full_attention"), "exact 40-layer"),
        (lambda c: setattr(c.model_config.hf_config, "sliding_window", 1024), "sliding_window=512"),
    ],
)
def test_hybrid_contract_fails_closed_on_any_sizing_input_drift(mutation, message, expect_error):
    config = _config()
    mutation(config)

    with expect_error(RuntimeError, message):
        hybrid_kv.validate_hybrid_kv_vllm_config(config)
