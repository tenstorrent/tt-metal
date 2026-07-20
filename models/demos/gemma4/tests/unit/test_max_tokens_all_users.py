# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the Gemma4-31B all-user KV-cache pool cap (``gemma4_max_tokens_all_users``).

The helper is vllm-free, so these run in the Gemma-4 models-unit-tests pipeline
(no device needed — device detection is monkeypatched).
"""

import pytest

from models.demos.gemma4.tt import common as gemma4_common
from models.demos.gemma4.tt.common import gemma4_max_tokens_all_users


def _as_wormhole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemma4_common, "is_wormhole_b0", lambda: True)
    monkeypatch.setattr(gemma4_common, "is_blackhole", lambda: False)


def _as_blackhole(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gemma4_common, "is_wormhole_b0", lambda: False)
    monkeypatch.setattr(gemma4_common, "is_blackhole", lambda: True)


def test_31b_capped_on_wormhole_t3k(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemma4-31B on WH-T3K (8 devices) caps the all-user pool at 32768."""
    _as_wormhole(monkeypatch)
    assert gemma4_max_tokens_all_users("google/gemma-4-31B-it", num_devices=8, tt_data_parallel=1) == 32_768


def test_31b_capped_on_blackhole_p150x4(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemma4-31B on BH-QB2/P150x4 (4 devices) caps the all-user pool at 32768."""
    _as_blackhole(monkeypatch)
    assert gemma4_max_tokens_all_users("google/gemma-4-31B-it", num_devices=4, tt_data_parallel=1) == 32_768


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """GEMMA4_MAX_TOKENS_ALL_USERS overrides the derived cap (per-deployment retune)."""
    _as_blackhole(monkeypatch)
    monkeypatch.setenv("GEMMA4_MAX_TOKENS_ALL_USERS", "16384")
    assert gemma4_max_tokens_all_users("google/gemma-4-31B-it", num_devices=4, tt_data_parallel=1) == 16_384


def test_non_31b_is_uncapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Smaller configs (e.g. 12B) get no Gemma4-specific cap (None => generic default)."""
    _as_blackhole(monkeypatch)
    assert gemma4_max_tokens_all_users("google/gemma-4-12B-it", num_devices=4, tt_data_parallel=1) is None


def test_31b_unmatched_device_count_is_uncapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """31B on a device layout the cap wasn't validated for is not capped."""
    _as_wormhole(monkeypatch)
    # WH but 4 devices — not the validated T3K(8) config
    assert gemma4_max_tokens_all_users("google/gemma-4-31B-it", num_devices=4, tt_data_parallel=1) is None
