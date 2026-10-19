# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The tuned AGMM overrides against the device worker grid (CPU only)."""

import pytest

from models.demos.pplx_decider_v1_27b.tt.prefill_tuning import AGMM_CONFIGS, make_agmm_config_fn

P150_GRID = (11, 10)  # harvested P150 (QuietBox 2)
GALAXY_GRID = (12, 10)  # Blackhole Galaxy chip, where the tables were tuned


@pytest.fixture(autouse=True)
def _default_tuning(monkeypatch):
    for var in ("PPLX_AGMM_TUNED", "PPLX_AGMM_WIDE"):
        monkeypatch.delenv(var, raising=False)


def _overrides(tp, grid):
    fn = make_agmm_config_fn(tp, grid)
    return {key: fn(key[1], key[2]) for key in AGMM_CONFIGS if key[0] == tp}


@pytest.mark.parametrize("tp", [4, 8])
def test_overrides_fit_a_p150_grid(tp):
    for key, cfg in _overrides(tp, P150_GRID).items():
        if cfg is not None:
            assert cfg.get("grid_x", 8) <= P150_GRID[0] and cfg.get("grid_y", 0) <= P150_GRID[1], f"{key}: {cfg}"


def test_wide_tp4_mlp_entry_falls_back_on_a_p150():
    assert make_agmm_config_fn(4, P150_GRID)(2048, "mlp_gateup") is None
    assert make_agmm_config_fn(4, GALAXY_GRID)(2048, "mlp_gateup")["grid_x"] == 12


@pytest.mark.parametrize("tp", [4, 8])
def test_galaxy_keeps_every_entry(tp):
    assert _overrides(tp, GALAXY_GRID) == {key: cfg for key, cfg in AGMM_CONFIGS.items() if key[0] == tp}
