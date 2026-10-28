# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only test of the tuned AGMM lookup key (tp_common.agmm_tuned_config): K and N are tile-padded (x32) in the key,
so the GDN in-proj weight (width 2060) matches the table entry 2080. Needs ttnn importable (no device).

  cd <worktree> && PYTHONPATH=<worktree>:tt-metal/ttnn:tt-metal/tools tt-metal/python_env/bin/python -m pytest -q \
      -p no:cacheprovider --noconftest models/demos/blackhole/qwen36/tests/test_agmm_tuned_key_host.py
"""

import types

import pytest

from models.demos.blackhole.qwen36.tt import tp_common as tc


def _w(k, n):
    return types.SimpleNamespace(shape=(1, 1, k, n))


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    print("tp_common module:", tc.__file__)
    monkeypatch.setenv("QWEN36_PREFILL_AGMM_TUNED", "1")


@pytest.mark.parametrize(
    "swiglu,k,n,blocks",
    [
        (False, 5120, 2060, (11, 9, 8, 4, 8)),  # GDN in-proj, padded 2080
        (False, 5120, 2048, (11, 9, 8, 4, 16)),  # attention qkv
        (True, 5120, 4352, (12, 9, 8, 4, 16)),  # MLP gate|up swiglu
    ],
)
def test_tuned_hit(swiglu, k, n, blocks):
    r = tc.agmm_tuned_config(8192, _w(k, n), swiglu)
    assert r is not None
    cfg, links, workers = r
    gx = blocks[0]
    assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == blocks[2:]
    assert cfg.compute_with_storage_grid_size.x == gx and cfg.compute_with_storage_grid_size.y == blocks[1]
    assert links == 2 and workers == -(-gx // 2)


def test_small_rows_none():
    assert tc.agmm_tuned_config(2048, _w(5120, 2060), False) is None


def test_other_shape_none():
    assert tc.agmm_tuned_config(8192, _w(6144, 640), False) is None
