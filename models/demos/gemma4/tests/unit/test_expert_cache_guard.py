# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the expert loader refuses a warm-cache placeholder state dict when its tensorbins are missing."""
import ttnn
from models.demos.gemma4.tt.experts.weights import _refuse_placeholder_on_cache_miss


class _Placeholder(dict):
    is_placeholder = True


def _tensorbin(prefix):
    return f"{prefix}_dtype_{ttnn.bfloat8_b.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"


def test_real_state_dict_never_refused(tmp_path):
    _refuse_placeholder_on_cache_miss({}, [str(tmp_path / "gate_up_proj_x")], ttnn.bfloat8_b)


def test_placeholder_refused_when_tensorbin_missing(tmp_path, expect_error):
    with expect_error(RuntimeError, "cache miss"):
        _refuse_placeholder_on_cache_miss(_Placeholder(), [str(tmp_path / "gate_up_proj_x")], ttnn.bfloat8_b)


def test_placeholder_refused_without_cache_path(expect_error):
    with expect_error(RuntimeError, "cache miss"):
        _refuse_placeholder_on_cache_miss(_Placeholder(), [None], ttnn.bfloat8_b)


def test_placeholder_accepted_when_all_tensorbins_present(tmp_path):
    prefixes = [str(tmp_path / "gate_up_proj_x"), str(tmp_path / "down_proj_x")]
    for p in prefixes:
        open(_tensorbin(p), "wb").close()
    _refuse_placeholder_on_cache_miss(_Placeholder(), prefixes, ttnn.bfloat8_b)


def test_placeholder_refused_when_one_tensorbin_missing(tmp_path, expect_error):
    prefixes = [str(tmp_path / "gate_up_proj_x"), str(tmp_path / "down_proj_x")]
    open(_tensorbin(prefixes[0]), "wb").close()
    with expect_error(RuntimeError, "down_proj_x"):
        _refuse_placeholder_on_cache_miss(_Placeholder(), prefixes, ttnn.bfloat8_b)
