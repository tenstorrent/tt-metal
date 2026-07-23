# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for device-init latent helpers (no TT device required)."""

from __future__ import annotations

import torch
import pytest

from models.experimental.hunyuan_image_3_0.ref.model_config import (
    PRODUCTION_LATENT_GRID,
    VAE_LATENT_CHANNELS,
)
from models.experimental.hunyuan_image_3_0.tt.noise import resolve_latent_nchw


@pytest.mark.unit_host
def test_resolve_latent_nchw_torch():
    g = PRODUCTION_LATENT_GRID
    c = VAE_LATENT_CHANNELS
    x = torch.randn(1, c, g, g)
    assert resolve_latent_nchw(x, token_h=g, token_w=g) == (1, c, g, g)


@pytest.mark.unit_host
def test_resolve_latent_nchw_rejects_bad_rank(expect_error):
    g = PRODUCTION_LATENT_GRID
    c = VAE_LATENT_CHANNELS
    with expect_error(ValueError, "NCHW"):
        resolve_latent_nchw(torch.randn(c, g, g), token_h=g, token_w=g)
