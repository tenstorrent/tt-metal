# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from laguna_vllm_ext import prefix_cache


class _BasePrefixModel:
    model_capabilities = {"supports_prefix_caching": True}


class _SlidingPrefixModel:
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_prefix_caching_with_sliding_window": True,
    }


class _SlidingOnlyModel:
    model_capabilities = {"supports_prefix_caching_with_sliding_window": True}


def _config(*, requested: bool = True, sliding_window: int | None = 512):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=requested,
            block_size=64,
        ),
        model_config=SimpleNamespace(get_sliding_window=lambda: sliding_window),
        scheduler_config=SimpleNamespace(
            enable_chunked_prefill=False,
            max_num_seqs=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
    )


def _platform_that_disables_sliding_prefix_cache():
    class FakeTTPlatform:
        calls = 0

        @classmethod
        def check_and_update_config(cls, vllm_config):
            cls.calls += 1
            if (
                vllm_config.cache_config.enable_prefix_caching
                and vllm_config.model_config.get_sliding_window() is not None
            ):
                vllm_config.cache_config.enable_prefix_caching = False

    return FakeTTPlatform


@pytest.fixture(autouse=True)
def _prefix_cache_disabled_by_default(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_PREFIX_CACHE", "0")


@pytest.mark.parametrize(
    ("model_class", "expected_enabled", "expected_restored"),
    [
        (_SlidingPrefixModel, True, True),
        (_BasePrefixModel, False, False),
        (_SlidingOnlyModel, False, False),
        (type("NoCapabilitiesModel", (), {}), False, False),
    ],
)
def test_only_explicit_dual_capability_restores_prefix_cache(
    caplog, monkeypatch, model_class, expected_enabled, expected_restored
):
    caplog.set_level("INFO", logger=prefix_cache.__name__)
    platform_class = _platform_that_disables_sliding_prefix_cache()
    monkeypatch.setattr(
        prefix_cache, "_resolve_model_class", lambda _config: model_class
    )

    assert prefix_cache._patch_platform(platform_class)
    config = _config()
    platform_class.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is expected_enabled
    assert platform_class.calls == 1
    assert (
        "Laguna prefix-cache final state: "
        f"requested=True enabled={expected_enabled} "
        f"restored_by_capability={expected_restored}"
    ) in caplog.messages


def test_does_not_enable_prefix_cache_that_was_not_requested(caplog, monkeypatch):
    caplog.set_level("INFO", logger=prefix_cache.__name__)
    platform_class = _platform_that_disables_sliding_prefix_cache()
    monkeypatch.setattr(
        prefix_cache,
        "_resolve_model_class",
        lambda _config: pytest.fail("model resolution must stay lazy"),
    )

    prefix_cache._patch_platform(platform_class)
    config = _config(requested=False)
    platform_class.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is False
    assert platform_class.calls == 1
    assert (
        "Laguna prefix-cache final state: requested=False enabled=False "
        "restored_by_capability=False"
    ) in caplog.messages


def test_non_sliding_model_is_unchanged(caplog, monkeypatch):
    caplog.set_level("INFO", logger=prefix_cache.__name__)
    platform_class = _platform_that_disables_sliding_prefix_cache()
    monkeypatch.setattr(
        prefix_cache,
        "_resolve_model_class",
        lambda _config: pytest.fail("model resolution must stay lazy"),
    )

    prefix_cache._patch_platform(platform_class)
    config = _config(sliding_window=None)
    platform_class.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is True
    assert platform_class.calls == 1
    assert (
        "Laguna prefix-cache final state: requested=True enabled=True "
        "restored_by_capability=False"
    ) in caplog.messages


def test_platform_patch_is_idempotent(monkeypatch):
    platform_class = _platform_that_disables_sliding_prefix_cache()
    monkeypatch.setattr(
        prefix_cache, "_resolve_model_class", lambda _config: _SlidingPrefixModel
    )

    assert prefix_cache._patch_platform(platform_class)
    assert not prefix_cache._patch_platform(platform_class)
    config = _config()
    platform_class.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is True
    assert platform_class.calls == 1


def test_enabled_quantum_policy_is_checked_after_capability_restore(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_PREFIX_CACHE", "1")
    monkeypatch.setenv("TT_LAGUNA_PREFILL_FAST_CHUNK", "8192")
    platform_class = _platform_that_disables_sliding_prefix_cache()
    monkeypatch.setattr(
        prefix_cache, "_resolve_model_class", lambda _config: _SlidingPrefixModel
    )
    prefix_cache._patch_platform(platform_class)

    config = _config()
    platform_class.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is True


def test_enabled_quantum_policy_rejects_chunked_scheduler_prefill(monkeypatch):
    monkeypatch.setenv("TT_LAGUNA_PREFIX_CACHE", "1")
    monkeypatch.setenv("TT_LAGUNA_PREFILL_FAST_CHUNK", "8192")
    platform_class = _platform_that_disables_sliding_prefix_cache()
    monkeypatch.setattr(
        prefix_cache, "_resolve_model_class", lambda _config: _SlidingPrefixModel
    )
    prefix_cache._patch_platform(platform_class)
    config = _config()
    config.scheduler_config.enable_chunked_prefill = True

    with pytest.raises(RuntimeError, match="chunked prefill"):
        platform_class.check_and_update_config(config)
