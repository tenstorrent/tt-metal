# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4.0 Euler flow-matching sampler. Two checks:
#   1. host: the tt sampler's logit-normal schedule, step intervals, guidance
#      weights and preset table match the OFFICIAL reference scheduler exactly.
#   2. device: a full Euler step loop z <- z + v*(s-t) matches a torch reference.
# =============================================================================

import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.ideogram4.sampler import PRESETS, Ideogram4Sampler
from ....reference.ideogram4 import sampler_configs as ref_cfg
from ....reference.ideogram4 import scheduler as ref_sched
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor


@pytest.mark.parametrize("preset", list(PRESETS))
@pytest.mark.parametrize("resolution", [(1024, 1024), (2048, 768)], ids=["1024sq", "2048x768"])
def test_sampler_schedule_matches_reference(*, preset: str, resolution: tuple[int, int]) -> None:
    """Host-only: schedule, intervals, guidance weights match the reference bit-for-bit."""
    h, w = resolution
    params = PRESETS[preset]
    ref_params = ref_cfg.PRESETS[preset]

    assert params.num_steps == ref_params.num_steps
    assert tuple(params.guidance_schedule) == tuple(ref_params.guidance_schedule)
    assert params.mu == ref_params.mu and params.std == ref_params.std

    samp = Ideogram4Sampler.from_preset(preset, height=h, width=w)
    ref_schedule = ref_sched.get_schedule_for_resolution((h, w), known_mean=ref_params.mu, std=ref_params.std)
    ref_intervals = ref_sched.make_step_intervals(ref_params.num_steps)

    for i in range(params.num_steps):
        t_val, s_val = samp.times_for_step(i)
        ref_t = float(ref_schedule(ref_intervals[i + 1].unsqueeze(0)).item())
        ref_s = float(ref_schedule(ref_intervals[i].unsqueeze(0)).item())
        assert abs(t_val - ref_t) < 1e-6, f"step {i}: t {t_val} != {ref_t}"
        assert abs(s_val - ref_s) < 1e-6, f"step {i}: s {s_val} != {ref_s}"
        assert samp.guidance_weight(i) == float(ref_params.guidance_schedule[i])

    logger.info(f"sampler schedule OK: {preset} @ {resolution}, {params.num_steps} steps")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
@pytest.mark.parametrize("preset", ["V4_TURBO_12"])
def test_sampler_euler_step_device(*, mesh_device: ttnn.MeshDevice, preset: str) -> None:
    """Device: the full Euler step loop matches torch z <- z + v*(s-t)."""
    torch.manual_seed(0)
    h = w = 1024
    samp = Ideogram4Sampler.from_preset(preset, height=h, width=w)

    batch, tokens, ch = 1, 4096, 128
    z = torch.randn(batch, tokens, ch, dtype=torch.float32)
    tt_z = bf16_tensor(z, device=mesh_device)

    for i in reversed(range(samp.num_steps)):
        v = torch.randn(batch, tokens, ch, dtype=torch.float32)  # stand-in velocity
        t_val, s_val = samp.times_for_step(i)
        z = z + v * (s_val - t_val)  # torch reference
        tt_z = samp.step(tt_z, bf16_tensor(v, device=mesh_device), i)

    tt_z_torch = tensor.to_torch(tt_z, mesh_axes=[None, None, None])
    logger.info(f"sampler device step OK: {preset}, {samp.num_steps} steps")
    assert_quality(z, tt_z_torch, pcc=0.999)
