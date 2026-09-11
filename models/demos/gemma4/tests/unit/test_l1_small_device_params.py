# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: L1_SMALL defaults so CCL all_gather semaphores do not fragment L1."""

from ...tt.ccl import default_l1_small_size
from ..test_factory import with_l1_small


def test_default_l1_small_size(monkeypatch):
    monkeypatch.delenv("GEMMA4_L1_SMALL_SIZE", raising=False)
    assert default_l1_small_size() == 24576
    monkeypatch.setenv("GEMMA4_L1_SMALL_SIZE", "32768")
    assert default_l1_small_size() == 32768


def test_with_l1_small_sets_default(monkeypatch):
    monkeypatch.delenv("GEMMA4_L1_SMALL_SIZE", raising=False)
    params = with_l1_small({"fabric_config": None, "trace_region_size": 1})
    assert params["l1_small_size"] == default_l1_small_size()
    assert params["trace_region_size"] == 1


def test_with_l1_small_preserves_override():
    params = with_l1_small({"l1_small_size": 4096, "trace_region_size": 1})
    assert params["l1_small_size"] == 4096
    assert params["trace_region_size"] == 1
