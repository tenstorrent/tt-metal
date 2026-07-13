# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for ``Gemma4ForCausalLM.get_max_tokens_all_users``.

The all-user KV-cache pool cap is a model/device constraint (hybrid KV groups
off => every layer allocates a full-length KV buffer, so the default ~131K pool
OOMs DRAM on the larger 31B configs). It lives in the model class rather than a
CI/deploy env var — see tt-metal #49745. These tests lock the derived values and
the override/fallback behavior; they call the classmethod directly, so no device
is required (device detection is monkeypatched).
"""

import pytest

from models.common.model_capabilities import FALLBACK_MAX_TOKENS_ALL_USERS
from models.demos.gemma4.tt import generator_vllm as generator_vllm_module

Gemma4ForCausalLM = generator_vllm_module.Gemma4ForCausalLM


@pytest.fixture(autouse=True)
def _hybrid_kv_groups_off(monkeypatch: pytest.MonkeyPatch):
    """Default config: hybrid KV groups disabled (the state the cap is defined for)."""
    monkeypatch.setattr(Gemma4ForCausalLM, "_HYBRID_KV_CACHE_GROUPS_ENABLED", False)


def _as_wormhole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_vllm_module, "is_wormhole_b0", lambda: True)
    monkeypatch.setattr(generator_vllm_module, "is_blackhole", lambda: False)


def _as_blackhole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generator_vllm_module, "is_wormhole_b0", lambda: False)
    monkeypatch.setattr(generator_vllm_module, "is_blackhole", lambda: True)


def test_31b_capped_on_wormhole_t3k(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemma4-31B on WH-T3K (8 devices) caps the all-user pool at 32768."""
    _as_wormhole(monkeypatch)
    assert (
        Gemma4ForCausalLM.get_max_tokens_all_users(
            model_name="google/gemma-4-31B-it",
            num_devices=8,
            tt_data_parallel=1,
            max_model_len=32_768,
            max_num_seqs=1,
        )
        == 32_768
    )


def test_31b_capped_on_blackhole_p150x4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemma4-31B on BH-QB2/P150x4 (4 devices) caps the all-user pool at 32768."""
    _as_blackhole(monkeypatch)
    assert (
        Gemma4ForCausalLM.get_max_tokens_all_users(
            model_name="google/gemma-4-31B-it",
            num_devices=4,
            tt_data_parallel=1,
            max_model_len=32_768,
            max_num_seqs=1,
        )
        == 32_768
    )


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEMMA4_MAX_TOKENS_ALL_USERS overrides the derived cap (tt-inference-server retune)."""
    _as_blackhole(monkeypatch)
    monkeypatch.setenv("GEMMA4_MAX_TOKENS_ALL_USERS", "16384")
    assert (
        Gemma4ForCausalLM.get_max_tokens_all_users(
            model_name="google/gemma-4-31B-it",
            num_devices=4,
            tt_data_parallel=1,
        )
        == 16_384
    )


def test_non_31b_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smaller configs (e.g. 12B) are not capped and use the fallback capacity."""
    _as_blackhole(monkeypatch)
    assert (
        Gemma4ForCausalLM.get_max_tokens_all_users(
            model_name="google/gemma-4-12B-it",
            num_devices=4,
            tt_data_parallel=1,
        )
        == FALLBACK_MAX_TOKENS_ALL_USERS
    )


def test_31b_unmatched_device_count_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """31B on a device layout the cap wasn't validated for defers to the fallback."""
    _as_wormhole(monkeypatch)
    assert (
        Gemma4ForCausalLM.get_max_tokens_all_users(
            model_name="google/gemma-4-31B-it",
            num_devices=4,  # WH but 4 devices — not the validated T3K(8) config
            tt_data_parallel=1,
        )
        == FALLBACK_MAX_TOKENS_ALL_USERS
    )


def test_cap_lifts_when_hybrid_kv_groups_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With hybrid KV groups on, sliding layers allocate only their window, so the cap lifts."""
    _as_blackhole(monkeypatch)
    monkeypatch.setattr(Gemma4ForCausalLM, "_HYBRID_KV_CACHE_GROUPS_ENABLED", True)
    assert (
        Gemma4ForCausalLM.get_max_tokens_all_users(
            model_name="google/gemma-4-31B-it",
            num_devices=4,
            tt_data_parallel=1,
        )
        == FALLBACK_MAX_TOKENS_ALL_USERS
    )
