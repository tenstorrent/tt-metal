# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest


def _default_tt_dit_cache_dir(*, is_ci_env: bool) -> str:
    if is_ci_env:
        return "/tmp/TT_DIT_CACHE"
    user = os.environ.get("USER", "unknown")
    return f"/proj_sw/user_dev/{user}/tt_dit_cache"


@pytest.fixture
def tt_dit_cache_dir(monkeypatch: pytest.MonkeyPatch, is_ci_env: bool) -> None:
    if os.getenv("TT_DIT_CACHE_DIR") is None:
        monkeypatch.setenv("TT_DIT_CACHE_DIR", _default_tt_dit_cache_dir(is_ci_env=is_ci_env))
