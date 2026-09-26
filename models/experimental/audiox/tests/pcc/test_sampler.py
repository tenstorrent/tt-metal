# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.audiox.reference.sampler import sample_discrete_euler as ref_sample, sample_rf as ref_sample_rf
from models.experimental.audiox.tt.sampler import sample_discrete_euler as tt_sample, sample_rf as tt_sample_rf
from tests.ttnn.utils_for_testing import assert_with_pcc


PCC_THRESHOLD = 0.99


def _ref_velocity_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Deterministic mock denoiser: scales x by mean(t). Mirrors the TT mock so
    both samplers consume the same effective velocity field."""
    scale = t.mean()
    return x * scale


def _tt_velocity_model(x: ttnn.Tensor, t: ttnn.Tensor) -> ttnn.Tensor:
    scale = float(ttnn.to_torch(t).mean().item())
    return ttnn.multiply(x, scale)


@pytest.mark.parametrize("batch, channels, length, steps", [(1, 4, 64, 8)])
def test_sample_discrete_euler_pcc(device, batch, channels, length, steps):
    torch.manual_seed(0)
    noise = torch.randn(batch, channels, length)

    ref_out = ref_sample(_ref_velocity_model, noise.clone(), steps=steps, sigma_max=1.0)

    tt_noise = ttnn.from_torch(noise, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_sample(_tt_velocity_model, tt_noise, steps=steps, mesh_device=device, sigma_max=1.0)

    assert_with_pcc(ref_out, ttnn.to_torch(tt_out), pcc=PCC_THRESHOLD)


@pytest.mark.parametrize("batch, channels, length, steps, sigma_max", [(1, 4, 64, 8, 0.7)])
def test_sample_rf_with_init_data_pcc(device, batch, channels, length, steps, sigma_max):
    torch.manual_seed(1)
    noise = torch.randn(batch, channels, length)
    init = torch.randn(batch, channels, length)

    ref_out = ref_sample_rf(
        _ref_velocity_model, noise.clone(), init_data=init.clone(), steps=steps, sigma_max=sigma_max
    )

    tt_noise = ttnn.from_torch(noise, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_init = ttnn.from_torch(init, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = tt_sample_rf(
        _tt_velocity_model,
        tt_noise,
        mesh_device=device,
        init_data=tt_init,
        steps=steps,
        sigma_max=sigma_max,
    )

    assert_with_pcc(ref_out, ttnn.to_torch(tt_out), pcc=PCC_THRESHOLD)
