# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TT_DIT_MM_REBLOCK: force a different-but-valid blocking on one matmul shape.

This is the noise-floor control. A blocking is a loop tiling of the same sum, so an override is
math-equivalent to the tuned config and differs only in floating-point reduction order — which is
what lets a run be numerically different from the baseline while being mathematically identical to
it. Host-only: the parse and the precedence are what can silently rot."""

import pytest

import ttnn
from models.tt_dit.utils.matmul import get_matmul_config, grid_12_9_configs


@pytest.fixture(autouse=True)
def _clear_reblock_env(monkeypatch):
    monkeypatch.delenv("TT_DIT_MM_REBLOCK", raising=False)


def _cfg(M, K, N, grid=(12, 9)):
    return get_matmul_config(M, K, N, ttnn.CoreCoord(*grid))


def test_no_env_leaves_the_tuned_config_untouched():
    cfg = _cfg(4864, 4096, 3072)
    assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == grid_12_9_configs[(4864, 4096, 3072)][:3]


def test_override_replaces_the_tuned_config(monkeypatch):
    monkeypatch.setenv("TT_DIT_MM_REBLOCK", "4864,4096,3072=5,8,16,1,4")
    cfg = _cfg(4864, 4096, 3072)
    assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == (5, 8, 16)
    assert (cfg.subblock_h, cfg.subblock_w) == (1, 4)


def test_override_is_scoped_to_its_shape(monkeypatch):
    """Only the named (M, K, N) moves; every other shape keeps its tuned entry."""
    monkeypatch.setenv("TT_DIT_MM_REBLOCK", "4864,4096,3072=5,8,16,1,4")
    other = _cfg(4864, 4096, 4096)
    assert (other.M_block_size, other.K_block_size, other.N_block_size) == grid_12_9_configs[(4864, 4096, 4096)][:3]


def test_override_applies_on_any_grid_and_to_untuned_shapes(monkeypatch):
    """Keyed on shape alone: it must fire even where the shape has no tuned entry (the fallback path),
    otherwise a control run could silently not be a control."""
    monkeypatch.setenv("TT_DIT_MM_REBLOCK", "1024,1024,1024=2,4,8")
    cfg = _cfg(1024, 1024, 1024, grid=(8, 8))
    assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == (2, 4, 8)


def test_multiple_entries(monkeypatch):
    monkeypatch.setenv("TT_DIT_MM_REBLOCK", "4864,4096,3072=5,8,16,1,4; 1216,4096,3072=4,4,12")
    assert _cfg(4864, 4096, 3072).K_block_size == 8
    assert _cfg(1216, 4096, 3072).K_block_size == 4


def test_env_change_is_picked_up_not_cached_from_a_previous_parse(monkeypatch):
    monkeypatch.setenv("TT_DIT_MM_REBLOCK", "4864,4096,3072=5,8,16,1,4")
    assert _cfg(4864, 4096, 3072).N_block_size == 16
    monkeypatch.setenv("TT_DIT_MM_REBLOCK", "4864,4096,3072=10,4,12,1,4")
    assert _cfg(4864, 4096, 3072).N_block_size == 12


def test_malformed_entry_raises(monkeypatch, expect_error):
    """A typo'd override must be loud: silently ignoring it would run a "control" that never fired."""
    monkeypatch.setenv("TT_DIT_MM_REBLOCK", "4864,4096=5,8,16")
    with expect_error(ValueError, "TT_DIT_MM_REBLOCK"):
        _cfg(4864, 4096, 3072)
