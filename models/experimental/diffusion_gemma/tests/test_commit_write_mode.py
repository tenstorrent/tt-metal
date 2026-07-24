# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the commit KV-write mode knob (#47557).

``tt/commit_batched.py`` selects the contiguous-cache KV-write mechanism from
``DG_COMMIT_KV_WRITE`` (``fill`` = one ``ttnn.fill_cache`` per K/V per layer, the default;
``position`` = the per-position ``paged_update_cache`` reference). The mechanisms
themselves are gated on device by ``test_device_commit_kv_write.py``.
"""

import importlib

import pytest

import models.experimental.diffusion_gemma.tt.commit_batched as commit_batched


def _resolve(monkeypatch, *, kv_write=None, write_batch=None):
    """Resolve the default mode under a given environment."""
    for name, value in (("DG_COMMIT_KV_WRITE", kv_write), ("DG_COMMIT_WRITE_BATCH", write_batch)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    return commit_batched._default_kv_write_mode()


def test_defaults_to_one_op_fill(monkeypatch):
    assert _resolve(monkeypatch) == "fill"


@pytest.mark.parametrize("value,expected", [("fill", "fill"), ("position", "position")])
def test_explicit_mode_wins(monkeypatch, value, expected):
    assert _resolve(monkeypatch, kv_write=value) == expected


@pytest.mark.parametrize("value,expected", [(" FILL ", "fill"), ("Position", "position")])
def test_mode_is_case_and_space_insensitive(monkeypatch, value, expected):
    assert _resolve(monkeypatch, kv_write=value) == expected


def test_obsolete_write_batch_is_ignored_not_honored(monkeypatch):
    """The removed 1-block-paged batch knob must not resurrect the slow path silently.

    A stale ``DG_COMMIT_WRITE_BATCH`` export used to select the mechanism; it is now
    ignored (the fast fill write still wins) and warned about.
    """
    assert _resolve(monkeypatch, write_batch="1") == "fill"
    assert _resolve(monkeypatch, write_batch="8") == "fill"
    # ...and it still cannot override an explicit choice.
    assert _resolve(monkeypatch, kv_write="position", write_batch="8") == "position"


def test_obsolete_write_batch_warns(monkeypatch, caplog):
    monkeypatch.delenv("DG_COMMIT_KV_WRITE", raising=False)
    monkeypatch.setenv("DG_COMMIT_WRITE_BATCH", "8")
    messages = []
    handler_id = commit_batched.logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        commit_batched._default_kv_write_mode()
    finally:
        commit_batched.logger.remove(handler_id)
    assert any("DG_COMMIT_WRITE_BATCH is obsolete" in m for m in messages)


def test_invalid_mode_fails_loudly(monkeypatch, expect_error):
    with expect_error(ValueError, match="DG_COMMIT_KV_WRITE"):
        _resolve(monkeypatch, kv_write="fastest")


def test_paged_mode_is_gone(monkeypatch, expect_error):
    """The 1-block-paged batched write was racy by construction and was removed."""
    assert "paged" not in commit_batched._KV_WRITE_MODES
    with expect_error(ValueError, match="DG_COMMIT_KV_WRITE"):
        _resolve(monkeypatch, kv_write="paged")


def test_module_default_is_fill_in_a_clean_env(monkeypatch):
    """Re-importing with no knobs set leaves the shipped default at ``fill``."""
    monkeypatch.delenv("DG_COMMIT_KV_WRITE", raising=False)
    monkeypatch.delenv("DG_COMMIT_WRITE_BATCH", raising=False)
    reloaded = importlib.reload(commit_batched)
    try:
        assert reloaded._DEFAULT_KV_WRITE_MODE == "fill"
    finally:
        importlib.reload(commit_batched)


def test_unknown_write_mode_argument_fails_loudly(expect_error):
    with expect_error(ValueError, match="write_mode must be one of"):
        commit_batched._write_canvas_kv_contiguous(
            None,
            None,
            _FakeTensor(),
            _FakeTensor(),
            start_pos=0,
            canvas_len=32,
            mesh_device=None,
            write_mode="turbo",
        )


class _FakeTensor:
    """Stand-in for a ttnn canvas tensor: only ``shape`` is read before the guard fires."""

    shape = (1, 2, 32, 256)
