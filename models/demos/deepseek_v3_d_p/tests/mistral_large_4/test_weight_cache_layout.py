# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Where the routed experts' TTNN weight cache lives, without a device."""

from pathlib import Path

import pytest

from models.demos.deepseek_v3_d_p.tt.mistral_large_4.experts import cache_complete
from models.demos.deepseek_v3_d_p.tt.mistral_large_4.hooks import experts_cache_dir
from models.demos.deepseek_v3_d_p.tt.runners.adapters.mistral_large_4 import (
    MistralLarge4PrefillAdapter,
    weight_cache_dir,
)


class _Spec:
    def __init__(self, cache):
        self._v = {"weights.cache": cache}

    def get(self, key, default=None):
        return self._v.get(key, default)


def test_runner_path_per_mesh(monkeypatch, tmp_path):
    monkeypatch.setenv("PREFILL_TTNN_CACHE", str(tmp_path))
    path = MistralLarge4PrefillAdapter().weight_cache_path((8, 4), sp_axis=0)
    assert path == tmp_path / "mistral_large_4_bh_32dev" / "8x4"
    assert not path.exists()  # created by the first run that writes it


def test_empty_root_means_no_cache(monkeypatch):
    monkeypatch.setenv("PREFILL_TTNN_CACHE", "")
    assert MistralLarge4PrefillAdapter().weight_cache_path((8, 4)) is None
    assert experts_cache_dir(_Spec(None), 0) is None


def test_sp_on_axis_1_refused():
    with pytest.raises(ValueError):
        weight_cache_dir("/cache", (4, 8), sp_axis=1)


def test_experts_dir_per_dtype_and_layer(monkeypatch):
    monkeypatch.setenv("MISTRAL_LARGE_4_EXPERTS_WEIGHT_DTYPE", "bfloat4_b")
    root = Path("/cache/mistral_large_4_bh_32dev/8x4")
    assert experts_cache_dir(_Spec(root), 7) == root / "experts_bfloat4_b" / "layer_07"


def test_complete_only_with_marker(tmp_path):
    assert not cache_complete(None)
    assert not cache_complete(tmp_path)
    (tmp_path / "done").touch()
    assert cache_complete(tmp_path)
