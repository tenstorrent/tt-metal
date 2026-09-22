# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: L1_SMALL is reserved where CCL semaphores land, and nowhere else.

The reservation is not free -- ttnn's DEFAULT_L1_SMALL_SIZE is 0, so taking it
costs 24 KB per core off the main L1 pool. These pin the scope so a gemma4 leg
on any other cluster, or on a single-device mesh, keeps opening as it does on
main.
"""

from ...tt.ccl import default_l1_small_size
from ..test_factory import with_l1_small


def test_default_l1_small_size(monkeypatch):
    monkeypatch.delenv("GEMMA4_L1_SMALL_SIZE", raising=False)
    assert default_l1_small_size() == 24576
    monkeypatch.setenv("GEMMA4_L1_SMALL_SIZE", "32768")
    assert default_l1_small_size() == 32768


def test_reserved_for_a_multi_device_t3k_mesh(monkeypatch):
    monkeypatch.delenv("GEMMA4_L1_SMALL_SIZE", raising=False)
    monkeypatch.setattr("models.demos.gemma4.tests.test_factory.is_t3k_cluster", lambda: True)
    params = with_l1_small({"fabric_config": None, "trace_region_size": 1}, mesh_shape=(1, 8))
    assert params["l1_small_size"] == default_l1_small_size()
    assert params["trace_region_size"] == 1


def test_not_reserved_on_a_single_device_mesh(monkeypatch):
    """A 1x1 mesh runs no collective, so it would be paying for nothing."""
    monkeypatch.setattr("models.demos.gemma4.tests.test_factory.is_t3k_cluster", lambda: True)
    params = with_l1_small({"fabric_config": None}, mesh_shape=(1, 1))
    assert "l1_small_size" not in params


def test_not_reserved_off_a_t3k(monkeypatch):
    """N150 / N300 / WH Galaxy / Blackhole keep ttnn's default of 0."""
    monkeypatch.setattr("models.demos.gemma4.tests.test_factory.is_t3k_cluster", lambda: False)
    for shape in ((1, 1), (1, 2), (1, 4), (1, 8), (1, 32)):
        assert "l1_small_size" not in with_l1_small({"fabric_config": None}, mesh_shape=shape), shape


def test_an_explicit_override_always_wins(monkeypatch):
    monkeypatch.setattr("models.demos.gemma4.tests.test_factory.is_t3k_cluster", lambda: True)
    params = with_l1_small({"l1_small_size": 4096, "trace_region_size": 1}, mesh_shape=(1, 8))
    assert params["l1_small_size"] == 4096
    assert params["trace_region_size"] == 1
