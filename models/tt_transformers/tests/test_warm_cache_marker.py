# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight (CPU-only, no device, no model download) tests for the warm ttnn weight-cache
detector generalized into tt_transformers (issue #45400, generalizes GPT-OSS PR #48531).

These exercise the *real* ModelArgs.weight_cache_is_complete / mark_weight_cache_complete
logic by binding the unbound methods to a tiny stub whose weight_cache_path points at a
tmp dir -- so we validate marker round-trip, staleness rejection, the force-load env
override, and the .tensorbin belt-and-suspenders check without constructing a full ModelArgs.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.tt_transformers.tt.model_config import ModelArgs

DTYPE = "bfp8"  # opaque here: the stub's weight_cache_path ignores it; marker stores str(dtype)


class _FakeArgs:
    """Minimal stand-in exposing exactly what the marker methods touch, with the real methods
    bound so the production logic is under test."""

    WEIGHT_CACHE_MARKER = ModelArgs.WEIGHT_CACHE_MARKER
    WEIGHT_CACHE_FORMAT_VERSION = ModelArgs.WEIGHT_CACHE_FORMAT_VERSION
    weight_cache_is_complete = ModelArgs.weight_cache_is_complete
    mark_weight_cache_complete = ModelArgs.mark_weight_cache_complete

    def __init__(self, cache_dir, model_name="Test-Model-8B", n_layers=32, mesh_shape=(1, 8)):
        self._cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.n_layers = n_layers
        self.dummy_weights = False
        self.mesh_device = SimpleNamespace(shape=mesh_shape)

    def weight_cache_path(self, dtype):
        return self._cache_dir


def _touch_tensorbin(cache_dir):
    (Path(cache_dir) / "some.weight.tensorbin").write_bytes(b"\x00")


@pytest.fixture(autouse=True)
def _clear_force_env(monkeypatch):
    monkeypatch.delenv("TT_TRANSFORMERS_FORCE_MODEL_LOAD", raising=False)


def test_cold_cache_is_incomplete(tmp_path):
    args = _FakeArgs(tmp_path)
    assert args.weight_cache_is_complete(DTYPE) is False


def test_mark_then_complete_roundtrip(tmp_path):
    args = _FakeArgs(tmp_path)
    args.mark_weight_cache_complete(DTYPE)
    _touch_tensorbin(tmp_path)  # a real build writes tensor files alongside the marker
    assert args.weight_cache_is_complete(DTYPE) is True

    # Sanity-check the marker payload.
    meta = json.loads((tmp_path / ModelArgs.WEIGHT_CACHE_MARKER).read_text())
    assert meta["model_name"] == "Test-Model-8B"
    assert meta["n_layers"] == 32
    assert meta["mesh_shape"] == "(1, 8)"
    assert meta["format_version"] == ModelArgs.WEIGHT_CACHE_FORMAT_VERSION


def test_marker_without_tensorbin_is_incomplete(tmp_path):
    args = _FakeArgs(tmp_path)
    args.mark_weight_cache_complete(DTYPE)
    # No .tensorbin present -> belt-and-suspenders check fails.
    assert args.weight_cache_is_complete(DTYPE) is False


def test_force_env_disables_skip(tmp_path, monkeypatch):
    args = _FakeArgs(tmp_path)
    args.mark_weight_cache_complete(DTYPE)
    _touch_tensorbin(tmp_path)
    assert args.weight_cache_is_complete(DTYPE) is True  # warm...
    monkeypatch.setenv("TT_TRANSFORMERS_FORCE_MODEL_LOAD", "1")
    assert args.weight_cache_is_complete(DTYPE) is False  # ...but forced to cold-load


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param({"format_version": 999}, id="stale-format"),
        pytest.param({"model_name": "Other-Model"}, id="wrong-model"),
        pytest.param({"n_layers": 1}, id="partial-build"),
        pytest.param({"mesh_shape": "(2, 4)"}, id="wrong-mesh"),
    ],
)
def test_stale_marker_rejected(tmp_path, mutate):
    args = _FakeArgs(tmp_path)
    args.mark_weight_cache_complete(DTYPE)
    _touch_tensorbin(tmp_path)
    marker = tmp_path / ModelArgs.WEIGHT_CACHE_MARKER
    meta = json.loads(marker.read_text())
    meta.update(mutate)
    marker.write_text(json.dumps(meta))
    assert args.weight_cache_is_complete(DTYPE) is False


def test_corrupt_marker_is_incomplete(tmp_path):
    args = _FakeArgs(tmp_path)
    _touch_tensorbin(tmp_path)
    (tmp_path / ModelArgs.WEIGHT_CACHE_MARKER).write_text("{ not json")
    assert args.weight_cache_is_complete(DTYPE) is False
