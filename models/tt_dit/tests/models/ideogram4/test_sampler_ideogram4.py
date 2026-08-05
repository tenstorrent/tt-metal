# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4.0 Euler flow-matching sampler. Two checks:
#   1. host: the preset table + logit-normal schedule / guidance weights are well-formed
#      (num_steps/guidance/mu/std match spec; times in [0,1], monotonic; gw per schedule).
#   2. device: a full Euler step loop z <- z + v*(s-t) matches a torch reference.
# =============================================================================

import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.ideogram4.sampler import PRESETS, Ideogram4Sampler
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

# Expected preset spec (num_steps, guidance schedule, logit-normal mean/std) — the
# official Ideogram 4.0 values, kept here as an independent regression against sampler.PRESETS.
_EXPECTED = {
    "V4_QUALITY_48": dict(num_steps=48, gw=(3.0,) * 3 + (7.0,) * 45, mu=0.0, std=1.5),
    "V4_DEFAULT_20": dict(num_steps=20, gw=(3.0,) * 2 + (7.0,) * 18, mu=0.0, std=1.75),
    "V4_TURBO_12": dict(num_steps=12, gw=(3.0,) * 1 + (7.0,) * 11, mu=0.5, std=1.75),
}


@pytest.mark.parametrize("preset", list(PRESETS))
@pytest.mark.parametrize("resolution", [(1024, 1024), (2048, 768)], ids=["1024sq", "2048x768"])
def test_sampler_schedule(*, preset: str, resolution: tuple[int, int]) -> None:
    """Host-only: preset table matches spec and the logit-normal schedule is well-formed."""
    h, w = resolution
    params = PRESETS[preset]
    exp = _EXPECTED[preset]

    assert params.num_steps == exp["num_steps"]
    assert tuple(float(g) for g in params.guidance_schedule) == exp["gw"]
    assert (params.mu, params.std) == (exp["mu"], exp["std"])

    samp = Ideogram4Sampler.from_preset(preset, height=h, width=w)
    times = [samp.times_for_step(i) for i in range(params.num_steps)]

    for i, (t_val, s_val) in enumerate(times):
        assert 0.0 <= t_val <= 1.0, f"step {i}: t {t_val} out of [0,1]"
        assert 0.0 <= s_val <= 1.0, f"step {i}: s {s_val} out of [0,1]"
        assert samp.guidance_weight(i) == float(params.guidance_schedule[i])

    # the schedule must sweep monotonically across the denoising loop
    t_seq = [t for t, _ in times]
    assert t_seq == sorted(t_seq) or t_seq == sorted(t_seq, reverse=True), f"non-monotonic schedule: {t_seq}"

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
