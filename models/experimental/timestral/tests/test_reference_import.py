# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests: the pinned upstream reference imports and the released checkpoints load strictly."""

from __future__ import annotations

import subprocess

import pytest
import torch

from models.experimental.timestral.common.timestral_src import (
    REFERENCE_COMMIT,
    REFERENCE_SRC,
    TIMESTRAL_VERSION,
    require_timestral,
)
from models.experimental.timestral.tests.golden_helpers import QUANTILE_KNOTS, load_checkpoint, log_golden, series

require_timestral()

from timestral import TimestralForecaster  # noqa: E402

# (d_model, heads, mlp_hidden, layers, variate layers, parameters) from the Hub configs.
CHECKPOINT_GEOMETRY = {
    "Timestral-5M": (192, 3, 512, 8, 2, None),
    "Timestral-209M": (768, 12, 2048, 24, 6, 209_312_256),
}


def test_reference_is_pinned_submodule():
    timestral = require_timestral()
    assert timestral.__version__ == TIMESTRAL_VERSION
    try:
        head = subprocess.run(
            ["git", "-C", str(REFERENCE_SRC.parent), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is unavailable for the reference submodule")
    assert head == REFERENCE_COMMIT, f"reference submodule is at {head}, the tests are pinned to {REFERENCE_COMMIT}"


@pytest.mark.parametrize("name", list(CHECKPOINT_GEOMETRY))
def test_checkpoint_loads_strict(name):
    model = load_checkpoint(name)
    d_model, heads, hidden, layers, variate_layers, params = CHECKPOINT_GEOMETRY[name]
    c = model.config
    assert (c.d_model, c.num_heads, c.mlp_hidden_dim, len(c.block_schedule)) == (d_model, heads, hidden, layers)
    assert sum(kind == "time_variate" for kind in c.block_schedule) == variate_layers
    assert c.patch_size == 32 and c.head_dim == 64 and c.rope_dim == 32
    assert c.quantile_knots == QUANTILE_KNOTS
    if params is not None:
        assert sum(p.numel() for p in model.parameters()) == params


def test_forecaster_smoke_5m():
    model = load_checkpoint("Timestral-5M")
    forecaster = TimestralForecaster(model)
    context = series(2, 1, 1000)
    context[0, 0, 100:140] = float("nan")
    out = forecaster.forecast(context, prediction_length=96)
    assert out.quantiles.shape == (2, 1, 96, len(QUANTILE_KNOTS))
    assert torch.isfinite(out.quantiles).all()
    assert (out.quantiles.diff(dim=-1) >= 0).all(), "quantiles must be sorted"
    log_golden("forecaster_5m/median", out.median)
