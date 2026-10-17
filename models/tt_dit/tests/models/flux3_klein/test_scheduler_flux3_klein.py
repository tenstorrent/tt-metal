# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the FLUX 3 Image Klein Euler schedule against the vendored reference ``sampling.py``:
the 17-point grid, the per-step (t, dt) the reference loop reads, the bf16 rounding of t the DiT sees, and the
dtype of the state update."""

import hashlib

import pytest
import torch

from ....pipelines.flux3_klein import scheduler as s
from ....reference.flux3_klein import sampling as ref_sampling

ALPHA = 6.93
# sha256[:16] of the fp32 17-point schedule; pinned from the vendored sampling.timesteps(16, 6.93).
GOLDEN_16 = "47b2e4ea9acca305"
# sampling.timesteps(16, 6.93) to 4 decimals (PLAN.md's "0.9900, 0.9800 ... 0.3161" were approximations).
SCHEDULE_16_4DP = [
    1.0000,
    0.9905,
    0.9798,
    0.9678,
    0.9541,
    0.9384,
    0.9203,
    0.8991,
    0.8739,
    0.8435,
    0.8061,
    0.7590,
    0.6979,
    0.6153,
    0.4975,
    0.3160,
    0.0000,
]


def _digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


def _reference_steps(schedule):
    """sampling.py:37-38 / :50-51 verbatim: what denoise / denoise_flow_map read per step."""
    return [(schedule[i].item(), (schedule[i + 1] - schedule[i]).item()) for i in range(len(schedule) - 1)]


@pytest.mark.parametrize("steps", [16, 50, 1, 4])
def test_schedule_matches_reference(steps):
    scheduler = s.Flux3KleinScheduler(steps, ALPHA)
    theirs = ref_sampling.timesteps(steps, ALPHA)
    assert scheduler.schedule.dtype == theirs.dtype == torch.float32
    assert torch.equal(scheduler.schedule, theirs)
    assert torch.equal(s.timesteps(steps, ALPHA), theirs)
    assert scheduler.schedule.numel() == steps + 1 and len(scheduler) == steps


def test_distilled_schedule_is_the_17_point_grid():
    scheduler = s.Flux3KleinScheduler()
    assert scheduler.num_steps == s.FLUX3_KLEIN_STEPS["distilled"] == 16 and scheduler.alpha == s.FLUX3_KLEIN_ALPHA
    grid = scheduler.schedule
    assert grid.numel() == 17
    assert grid[0].item() == 1.0 and grid[-1].item() == 0.0
    assert bool((grid[1:] < grid[:-1]).all()), "strictly decreasing, no deduplication"
    assert [round(v, 4) for v in grid.tolist()] == SCHEDULE_16_4DP
    assert _digest(grid) == GOLDEN_16


def test_per_step_values_match_reference_loop():
    scheduler = s.Flux3KleinScheduler()
    theirs = _reference_steps(scheduler.schedule)
    ours = scheduler.steps()
    assert len(ours) == len(theirs) == 16
    for step, (t, dt) in zip(ours, theirs):
        assert step.t == t and step.dt == dt, step
        assert step.t_next == scheduler.schedule[step.index + 1].item()
        assert step.dt < 0 and step.t > step.t_next
    assert ours[0].t == 1.0 and ours[-1].t_next == 0.0
    # First and last Euler step sizes of the distilled model, 4 dp.
    assert round(abs(ours[0].dt), 4) == 0.0095
    assert round(abs(ours[-1].dt), 4) == 0.3160
    assert sum(step.dt for step in ours) == pytest.approx(-1.0, abs=1e-6)


def test_step_index_bounds(expect_error):
    scheduler = s.Flux3KleinScheduler()
    with expect_error(IndexError, "step 16 outside 0..15"):
        scheduler.step(16)
    with expect_error(IndexError, "step -1 outside 0..15"):
        scheduler.step(-1)


def test_model_timestep_is_the_bf16_fill():
    """pipeline.py:136 allocates bf16 timesteps and sampling.py:39/:52 fill_(t) them: that is where t is rounded."""
    scheduler = s.Flux3KleinScheduler()
    rounded_any = False
    for step in scheduler.steps():
        rows = torch.zeros(5, dtype=torch.bfloat16)
        rows.fill_(step.t)
        assert step.t_model == rows[0].item()
        assert torch.equal(scheduler.model_timesteps(5, step), rows)
        # dit.py:202: the flow-map target is cast from the fp32 schedule entry.
        assert step.t_next_model == scheduler.schedule[step.index + 1].reshape(1).to(torch.bfloat16).item()
        # dit.py:43: 1000 * t still in bf16.
        assert step.embedding_t == (1000.0 * rows[:1]).item()
        rounded_any |= step.t_model != step.t
    assert rounded_any, "bf16 rounding must be visible on a 16-step schedule"
    assert s.model_timestep(1.0) == 1.0 and s.model_timestep(0.0) == 0.0
    assert scheduler.step(1).t_model == pytest.approx(0.9921875)  # 0.990472 rounded to bf16
    assert scheduler.step(1).embedding_t == 992.0


def test_update_has_the_reference_dtypes():
    """sampling.py:54: x (fp32) + dt * v (bf16): the product rounds to bf16, the sum is fp32."""
    scheduler = s.Flux3KleinScheduler()
    step = scheduler.step(3)
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(64, 128, generator=generator).to(torch.bfloat16).float()
    v = torch.randn(64, 128, generator=generator).to(torch.bfloat16)
    out = scheduler.update(x, v, step)
    assert out.dtype == torch.float32
    assert torch.equal(out, x + step.dt * v)
    assert (step.dt * v).dtype == torch.bfloat16
    assert not torch.equal(out, x + step.dt * v.float()), "an fp32 product would be a different number"


def test_update_follows_the_reference_flow_map_loop():
    """A full 16-step Euler walk with a stand-in DiT equals denoise_flow_map's arithmetic."""
    scheduler = s.Flux3KleinScheduler()
    generator = torch.Generator().manual_seed(11)
    x_ref = torch.randn(16, 128, generator=generator).to(torch.bfloat16).float()
    x_ours = x_ref.clone()
    velocities = [torch.randn(16, 128, generator=generator).to(torch.bfloat16) for _ in range(16)]
    schedule = ref_sampling.timesteps(16, ALPHA)
    for i in range(len(schedule) - 1):
        dt = (schedule[i + 1] - schedule[i]).item()
        x_ref = x_ref + dt * velocities[i]
        x_ours = scheduler.update(x_ours, velocities[i], scheduler.step(i))
    assert torch.equal(x_ours, x_ref)


def test_cfg_velocity_matches_reference_expression():
    generator = torch.Generator().manual_seed(5)
    unconditional = torch.randn(8, 128, generator=generator).to(torch.bfloat16)
    conditional = torch.randn(8, 128, generator=generator).to(torch.bfloat16)
    guidance = s.FLUX3_KLEIN_GUIDANCE["base"]
    assert guidance == 3.5 and s.FLUX3_KLEIN_GUIDANCE["distilled"] is None
    out = s.Flux3KleinScheduler.cfg_velocity(unconditional, conditional, guidance)
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, unconditional + guidance * (conditional - unconditional))  # sampling.py:41


def test_base_schedule_has_51_points():
    scheduler = s.Flux3KleinScheduler(s.FLUX3_KLEIN_STEPS["base"])
    assert scheduler.schedule.numel() == 51
    assert torch.equal(scheduler.schedule, ref_sampling.timesteps(50, ALPHA))
    assert torch.equal(scheduler.schedule[[0, -1]], torch.tensor([1.0, 0.0]))


def test_rejects_bad_parameters(expect_error):
    with expect_error(ValueError, "num_steps must be at least 1"):
        s.Flux3KleinScheduler(0)
    with expect_error(ValueError, "alpha must be positive"):
        s.Flux3KleinScheduler(16, 0.0)
