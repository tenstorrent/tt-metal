# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for device-init latent helpers (no TT device required)."""

from __future__ import annotations

import torch
import pytest

from models.experimental.hunyuan_image_3_0.tt.noise import resolve_latent_nchw


@pytest.mark.unit_host
def test_resolve_latent_nchw_torch():
    x = torch.randn(1, 32, 64, 64)
    assert resolve_latent_nchw(x, token_h=64, token_w=64) == (1, 32, 64, 64)


@pytest.mark.unit_host
def test_resolve_latent_nchw_rejects_bad_rank(expect_error):
    with expect_error(ValueError, "NCHW"):
        resolve_latent_nchw(torch.randn(32, 64, 64), token_h=64, token_w=64)
