# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the prefill SDPA program config (tt/attention/operations.py): every grid derives from the
device's compute grid (the CI Blackhole SKUs are harvested below the Galaxy's 13x10 and threw on a constant (12,10)),
the wide config for >= 8 Q heads per chip only where the grid admits it, the fixed configs otherwise, the env
overrides and the diagnostic knobs. No device.

    pytest models/demos/gemma4/tests/unit/test_prefill_sdpa_config_host.py --noconftest -p no:cacheprovider
"""

import pytest

from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.attention import operations as ops

KNOBS = (
    "GEMMA4_PREFILL_SDPA_WIDE",
    "GEMMA4_PREFILL_SDPA_QCHUNK",
    "GEMMA4_PREFILL_SDPA_KCHUNK",
    "GEMMA4_PREFILL_SDPA_GRID",
    "GEMMA4_PREFILL_SDPA_EXP_APPROX",
    "GEMMA4_PREFILL_SDPA_PROGCFG",
    "GEMMA4_TEST_GRID",
)
# The wide / clamped grids are a Blackhole matter (the wide config only forms there); on a Wormhole runner every grid
# collapses to the 8x8 class, so the unit job there runs the two grids that can differ (wh_n150 tier-3 step budget).
GRIDS = ((13, 10), (12, 10), (11, 10), (10, 10), (8, 8), (7, 8)) if is_blackhole() else ((8, 8), (7, 8))


def _grid(pc):
    return (pc.compute_with_storage_grid_size.x, pc.compute_with_storage_grid_size.y)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("device_grid", GRIDS, ids=[f"{x}x{y}" for x, y in GRIDS])
@pytest.mark.parametrize("heads", (4, 8, 16))
@pytest.mark.parametrize("head_dim", (256, 512))
def test_every_config_fits_the_device_grid(device_grid, heads, head_dim):
    monkeypatch_env = {}
    pc = ops.prefill_sdpa_program_config(head_dim, 2048, sliding_window=None, num_heads=heads, device_grid=device_grid)
    if pc is None:  # narrow chips on Blackhole take the op defaults
        assert is_blackhole() and heads < ops.PREFILL_SDPA_WIDE_MIN_HEADS and head_dim <= 256
        return
    gx, gy = _grid(pc)
    assert gx * gy <= device_grid[0] * device_grid[1] and gx <= device_grid[0] and gy <= device_grid[1]
    if head_dim >= 512:
        assert (gx, gy) == (min(8, device_grid[0]), min(4, device_grid[1])) and pc.q_chunk_size == 128
    elif heads >= ops.PREFILL_SDPA_WIDE_MIN_HEADS and is_blackhole() and device_grid[0] * device_grid[1] > 64:
        assert (gx, gy) == (min(12, device_grid[0]), min(10, device_grid[1])) and pc.q_chunk_size == 128  # wide
    else:
        assert (gx, gy) == (min(8, device_grid[0]), min(8, device_grid[1])) and pc.q_chunk_size == 256
    assert pc.k_chunk_size == 128 and pc.exp_approx_mode is False
    assert not monkeypatch_env


def test_unknown_device_grid_is_never_wide():
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8)
    assert _grid(pc) == (8, 8) and pc.q_chunk_size == 256
    pc = ops.prefill_sdpa_program_config(512, 2048, sliding_window=None, num_heads=8)
    assert _grid(pc) == (8, 4) and pc.q_chunk_size == 128


def test_wide_on_the_galaxy_grid_and_the_12b_defaults(monkeypatch):
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8, device_grid=(13, 10))
    if is_blackhole():
        assert _grid(pc) == (12, 10) and pc.q_chunk_size == 128
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=4, device_grid=(13, 10))
    if is_blackhole():
        assert pc is None
        monkeypatch.setenv("GEMMA4_PREFILL_SDPA_PROGCFG", "1")  # opt back in to the fixed config
        pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=4, device_grid=(13, 10))
    assert _grid(pc) == (8, 8) and (pc.q_chunk_size, pc.k_chunk_size) == (256, 128)
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_PROGCFG", "0")  # the op defaults everywhere
    assert ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8, device_grid=(13, 10)) is None


def test_window_cap_and_short_sequences():
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=128, num_heads=8, device_grid=(13, 10))
    assert pc.k_chunk_size == 64
    pc = ops.prefill_sdpa_program_config(256, 64, sliding_window=None, num_heads=8, device_grid=(13, 10))
    assert pc.q_chunk_size == 64 and pc.k_chunk_size == 64


def test_env_overrides_win_but_stay_on_the_device(monkeypatch):
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_WIDE", "0")
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8, device_grid=(13, 10))
    assert _grid(pc) == (8, 8) and pc.q_chunk_size == 256
    monkeypatch.delenv("GEMMA4_PREFILL_SDPA_WIDE")
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_QCHUNK", "64")
    monkeypatch.setenv("GEMMA4_PREFILL_SDPA_GRID", "12x10")
    pc = ops.prefill_sdpa_program_config(256, 2048, sliding_window=None, num_heads=8, device_grid=(10, 10))
    assert pc.q_chunk_size == 64 and _grid(pc) == (10, 10)  # the env grid is clamped to the device
