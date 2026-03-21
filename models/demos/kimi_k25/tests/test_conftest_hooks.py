# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""test_conftest_hooks.py — CPU meta-tests for the Kimi K2.5 conftest hooks.

These tests verify that ``conftest.pytest_configure`` and
``conftest.pytest_collection_modifyitems`` behave correctly **without**
running a nested pytest process.  They exercise the hook functions directly
using mock pytest objects, ensuring:

* ``pytest_configure`` registers the ``requires_device`` marker.
* ``pytest_collection_modifyitems`` skips device-gated tests when
  ``MESH_DEVICE`` is *not set* (CPU-only environment).
* ``pytest_collection_modifyitems`` skips tests when ``MESH_DEVICE`` is set
  to a *non-matching* device type.
* ``pytest_collection_modifyitems`` does **not** skip tests when
  ``MESH_DEVICE`` matches the marker's device type.
* ``pytest_collection_modifyitems`` leaves tests **without** a
  ``requires_device`` marker completely untouched.
* The hook never calls ``pytest.exit()`` — unlike the DSV3 hook, which
  aborts the session when no hardware is found, this hook just silently
  skips, allowing CPU-only unit test runs.

Class
-----
TestPytestConfigureHook   — ``pytest_configure`` marker registration
TestCollectionModifyHook  — ``pytest_collection_modifyitems`` skip logic
"""

from __future__ import annotations

import os
import inspect
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# pytest_configure
# ---------------------------------------------------------------------------


class TestPytestConfigureHook:
    """CPU-only: validate pytest_configure registers requires_device marker."""

    def test_hook_exists_in_conftest(self):
        """conftest module must export a ``pytest_configure`` callable."""
        import models.demos.kimi_k25.conftest as _conftest

        assert callable(getattr(_conftest, "pytest_configure", None)), (
            "conftest.pytest_configure must exist and be callable"
        )

    def test_requires_device_marker_registered(self):
        """pytest_configure must register the ``requires_device`` marker."""
        from models.demos.kimi_k25.conftest import pytest_configure

        mock_config = mock.MagicMock()
        pytest_configure(mock_config)

        # addinivalue_line must have been called with a "markers" entry that
        # contains "requires_device"
        calls = mock_config.addinivalue_line.call_args_list
        assert any(
            "markers" in str(c) and "requires_device" in str(c)
            for c in calls
        ), (
            "pytest_configure must call config.addinivalue_line with a "
            "'markers' entry containing 'requires_device'"
        )

    def test_marker_source_mentions_device_types(self):
        """The ``requires_device`` marker docstring must mention device types."""
        from models.demos.kimi_k25.conftest import pytest_configure

        src = inspect.getsource(pytest_configure)
        assert "TG" in src or "device_types" in src, (
            "pytest_configure source must mention supported device types (TG, etc.) "
            "so operators know what values to pass"
        )


# ---------------------------------------------------------------------------
# pytest_collection_modifyitems
# ---------------------------------------------------------------------------


def _make_mock_item(device_types):
    """Return a mock pytest item with ``@pytest.mark.requires_device(device_types)``."""
    mock_item = mock.MagicMock()
    mock_marker = mock.MagicMock()
    mock_marker.args = (device_types,)
    mock_marker.kwargs = {}
    mock_item.get_closest_marker.return_value = mock_marker
    return mock_item


def _make_plain_item():
    """Return a mock pytest item with **no** ``requires_device`` marker."""
    mock_item = mock.MagicMock()
    mock_item.get_closest_marker.return_value = None  # no marker
    return mock_item


def _was_skip_added(mock_item: mock.MagicMock) -> bool:
    """Return True if ``item.add_marker`` was called with a skip marker."""
    for call in mock_item.add_marker.call_args_list:
        added = call.args[0] if call.args else None
        if added is not None and "skip" in str(added).lower():
            return True
    return False


class TestCollectionModifyHook:
    """CPU-only: validate pytest_collection_modifyitems skip logic."""

    def test_hook_exists_in_conftest(self):
        """conftest module must export a ``pytest_collection_modifyitems`` callable."""
        import models.demos.kimi_k25.conftest as _conftest

        assert callable(getattr(_conftest, "pytest_collection_modifyitems", None)), (
            "conftest.pytest_collection_modifyitems must exist and be callable"
        )

    def test_skips_when_mesh_device_unset(self, monkeypatch):
        """Device-gated tests must be skipped when ``MESH_DEVICE`` is not set."""
        monkeypatch.delenv("MESH_DEVICE", raising=False)

        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        item = _make_mock_item(["TG"])
        pytest_collection_modifyitems(config=mock.MagicMock(), items=[item])

        assert _was_skip_added(item), (
            "pytest_collection_modifyitems must add a skip marker to a "
            "requires_device test when MESH_DEVICE is not set"
        )

    def test_skips_when_mesh_device_does_not_match(self, monkeypatch):
        """Tests must be skipped when ``MESH_DEVICE`` is set to a non-matching type."""
        monkeypatch.setenv("MESH_DEVICE", "N300")

        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        item = _make_mock_item(["TG"])  # test requires TG, env is N300
        pytest_collection_modifyitems(config=mock.MagicMock(), items=[item])

        assert _was_skip_added(item), (
            "pytest_collection_modifyitems must skip a TG-gated test when "
            "MESH_DEVICE=N300 (non-matching)"
        )

    def test_does_not_skip_when_mesh_device_matches(self, monkeypatch):
        """Tests must NOT be skipped when ``MESH_DEVICE`` matches the marker."""
        monkeypatch.setenv("MESH_DEVICE", "TG")

        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        item = _make_mock_item(["TG"])
        pytest_collection_modifyitems(config=mock.MagicMock(), items=[item])

        assert not _was_skip_added(item), (
            "pytest_collection_modifyitems must NOT skip a TG-gated test when "
            "MESH_DEVICE=TG (matching)"
        )

    def test_case_insensitive_device_match(self, monkeypatch):
        """``MESH_DEVICE`` comparison must be case-insensitive (``tg`` == ``TG``)."""
        monkeypatch.setenv("MESH_DEVICE", "tg")  # lowercase

        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        item = _make_mock_item(["TG"])
        pytest_collection_modifyitems(config=mock.MagicMock(), items=[item])

        assert not _was_skip_added(item), (
            "MESH_DEVICE=tg must match a requires_device(['TG']) marker "
            "(comparison must be case-insensitive)"
        )

    def test_no_skip_added_to_plain_item(self, monkeypatch):
        """Tests without ``requires_device`` must not be touched."""
        monkeypatch.delenv("MESH_DEVICE", raising=False)

        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        item = _make_plain_item()
        pytest_collection_modifyitems(config=mock.MagicMock(), items=[item])

        item.add_marker.assert_not_called()

    def test_multiple_items_only_device_gated_skipped(self, monkeypatch):
        """When items list is mixed, only device-gated items get a skip marker."""
        monkeypatch.delenv("MESH_DEVICE", raising=False)

        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        hw_item = _make_mock_item(["TG"])
        cpu_item = _make_plain_item()

        pytest_collection_modifyitems(config=mock.MagicMock(), items=[hw_item, cpu_item])

        assert _was_skip_added(hw_item), "hardware-gated item must be skipped"
        cpu_item.add_marker.assert_not_called()  # CPU item must be untouched

    def test_hook_never_calls_pytest_exit(self, monkeypatch):
        """Hook must never call ``pytest.exit()`` — key difference from DSV3.

        The DSV3 conftest aborts the entire session when ``MESH_DEVICE`` is
        unset.  Kimi's hook must not do this — it should silently skip so
        that CPU-only unit test tiers work in CI without setting ``MESH_DEVICE``.
        """
        monkeypatch.delenv("MESH_DEVICE", raising=False)

        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        items = [_make_mock_item(["TG"]) for _ in range(5)]

        with mock.patch("pytest.exit") as mock_exit:
            pytest_collection_modifyitems(config=mock.MagicMock(), items=items)
            mock_exit.assert_not_called()

    def test_string_device_type_normalised_to_list(self, monkeypatch):
        """A bare string marker arg (e.g. ``requires_device("TG")``) must work.

        Some tests may use the shorthand ``@pytest.mark.requires_device("TG")``
        (a single string, not a list).  The hook must normalise this to a list
        before comparison.
        """
        monkeypatch.setenv("MESH_DEVICE", "TG")

        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        # Single string arg (not a list)
        mock_item = mock.MagicMock()
        mock_marker = mock.MagicMock()
        mock_marker.args = ("TG",)  # bare string, not ["TG"]
        mock_marker.kwargs = {}
        mock_item.get_closest_marker.return_value = mock_marker

        pytest_collection_modifyitems(config=mock.MagicMock(), items=[mock_item])

        assert not _was_skip_added(mock_item), (
            "requires_device('TG') (bare string) must match MESH_DEVICE=TG"
        )

    def test_source_does_not_call_pytest_exit(self):
        """Source inspection: ``pytest_collection_modifyitems`` must not call ``pytest.exit``."""
        from models.demos.kimi_k25.conftest import pytest_collection_modifyitems

        src = inspect.getsource(pytest_collection_modifyitems)
        assert "pytest.exit" not in src, (
            "conftest.pytest_collection_modifyitems must not call pytest.exit() — "
            "unlike DSV3, Kimi's hook must allow CPU-only test runs"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
