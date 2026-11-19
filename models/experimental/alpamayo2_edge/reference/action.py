# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-torch reference of the Alpamayo 2 Edge action path.

Ported from NVlabs alpamayo2-edge: the per-waypoint action input projection (`action_in_proj.py`), the Euler
flow-matching sampler (`diffusion/flow_matching.py`), the expert position ids and attention mask
(`expert_utils.py`), and the course-heading unicycle decoder
(`action_space/unicycle_accel_curvature{,_course}.py`, `action_space/utils.py`, `geometry/rotation.py`).
"""

import math
from typing import Callable

import torch
import torch.nn as nn

from models.experimental.alpamayo2_edge.reference.config import ActionConfig

# ---------------------------------------------------------------------------------------------------------------------
# Action input projection
# ---------------------------------------------------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Upstream `ActionInProj`'s RMSNorm: `text.RMSNorm` takes its weight at construction, this one is filled by
    `load_state_dict` from the checkpoint's `encoder.trunk.*` keys."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return output.type_as(x) * self.weight


class FourierEncoder(nn.Module):
    """`dim // 2` log-spaced frequencies in [1, max_freq]; scalar -> [sin | cos] * sqrt(2)."""

    def __init__(self, dim: int, max_freq: float):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even, got {dim}")
        self.out_dim = dim
        # The checkpoint does not store the frequencies; they are rebuilt from config.
        freqs = torch.logspace(0, math.log10(max_freq), steps=dim // 2)
        self.register_buffer("freqs", freqs[None, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        arg = x[..., None] * self.freqs * 2 * torch.pi
        return torch.cat([torch.sin(arg), torch.cos(arg)], -1) * math.sqrt(2)


class ActionInProj(nn.Module):
    """Noisy actions [B, N, action_dim] and flow time t [B, 1, 1] -> expert tokens [B, N, out_dim].

    Each action dim and the timestep get their own Fourier features; the concatenation goes through the MLP
    `encoder.trunk` and a LayerNorm. Module names match the checkpoint's `expert.action_in_proj.*`.
    """

    def __init__(self, acfg: ActionConfig, out_dim: int, weights: dict | None = None, dtype=torch.float32):
        super().__init__()
        self.sinus = nn.ModuleList(FourierEncoder(acfg.fourier_feats, acfg.max_freq) for _ in range(acfg.action_dim))
        self.timestep_fourier_encoder = FourierEncoder(acfg.fourier_feats, acfg.max_freq)
        in_features = acfg.fourier_feats * (acfg.action_dim + 1)
        hidden = acfg.in_proj_hidden
        layers: list[nn.Module] = [nn.Linear(in_features, hidden), nn.SiLU()]
        for i in range(acfg.in_proj_layers):
            last = i == acfg.in_proj_layers - 1
            layers += [_RMSNorm(hidden), nn.Linear(hidden, out_dim if last else hidden)]
            if not last:
                layers.append(nn.SiLU())
        self.encoder = nn.Module()
        self.encoder.trunk = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)
        if weights is not None:
            self.load_state_dict(weights, strict=True)
        self.to(dtype)
        self.requires_grad_(False)

    @property
    def in_features(self) -> int:
        return self.encoder.trunk[0].in_features

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        batch, n_waypoints, _ = x.shape
        x, t = x.float(), t.float()
        action_feats = torch.cat([encoder(x[:, :, i]) for i, encoder in enumerate(self.sinus)], dim=-1)
        timestep_feats = self.timestep_fourier_encoder(t[..., -1]).repeat(1, n_waypoints, 1)
        features = torch.cat((action_feats, timestep_feats), dim=-1).to(self.encoder.trunk[0].weight.dtype)
        encoded = self.encoder.trunk(features.flatten(0, 1)).reshape(batch, n_waypoints, -1)
        return self.norm(encoded)


# ---------------------------------------------------------------------------------------------------------------------
# Flow matching and expert conditioning
# ---------------------------------------------------------------------------------------------------------------------


def euler_sample(
    step_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], noise: torch.Tensor, steps: int
) -> torch.Tensor:
    """Integrate the flow from `noise` (t = 0) to actions (t = 1) with `steps` fixed Euler steps.

    `step_fn(x, t)` gets t as [B, 1, ..., 1] at the start of each step. t and dt come from `linspace(0, 1)` in fp32,
    so dt is not exactly 1 / steps; this matches `FlowMatching._euler`.
    """
    batch, n_dim = noise.shape[0], noise.ndim - 1
    time_steps = torch.linspace(0.0, 1.0, steps + 1, device=noise.device)
    x = noise
    for i in range(steps):
        dt = (time_steps[i + 1] - time_steps[i]).view(1, *[1] * n_dim).expand(batch, *[1] * n_dim)
        t_start = time_steps[i].view(1, *[1] * n_dim).expand(batch, *[1] * n_dim)
        x = x + dt * step_fn(x, t_start)
    return x


def expert_position_ids(offsets: torch.Tensor, rope_deltas: torch.Tensor, n: int) -> torch.Tensor:
    """[3, B, n] M-RoPE positions of the action tokens: they continue each row's text right after its anchor."""
    position_ids = torch.arange(n, device=offsets.device).expand(3, offsets.shape[0], n).clone()
    return position_ids + (rope_deltas + offsets[:, None]).to(position_ids.device)


def expert_mask(offsets: torch.Tensor, prefix_mask: torch.Tensor, kv_len: int, n: int = 64) -> torch.Tensor:
    """Additive fp32 mask [B, 1, n, kv_len + n] for the non-causal expert over the shared cache.

    Cache columns at or past a row's offset (text after its anchor) and prompt padding are masked; the n action
    tokens see each other.
    """
    batch = offsets.shape[0]
    mask = torch.zeros(batch, 1, n, kv_len + n, dtype=torch.float32, device=offsets.device)
    min_value = torch.finfo(mask.dtype).min
    invalid = torch.arange(kv_len, device=offsets.device)[None, :] >= offsets[:, None]
    mask[..., :kv_len].masked_fill_(invalid[:, None, None, :], min_value)
    padding = prefix_mask[:, None, None, :] == 0
    mask[..., : prefix_mask.shape[-1]] = torch.where(padding, min_value, mask[..., : prefix_mask.shape[-1]])
    return mask


# ---------------------------------------------------------------------------------------------------------------------
# Course-heading unicycle action space (decode only)
# ---------------------------------------------------------------------------------------------------------------------

# Course offset fade-in speeds [m/s] and the bound on a raw course offset [rad], from the course action space.
_COURSE_FADE_START_MPS = 0.5
_COURSE_FADE_END_MPS = 1.0
_MAX_COURSE_OFFSET_RAD = 0.05


def _unwrap_angle(phi: torch.Tensor) -> torch.Tensor:
    d = torch.diff(phi, dim=-1)
    d = torch.atan2(torch.sin(d), torch.cos(d))
    return torch.cat([phi[..., :1], phi[..., :1] + torch.cumsum(d, dim=-1)], dim=-1)


def _so3_to_yaw(rot: torch.Tensor) -> torch.Tensor:
    return torch.atan2(rot[..., 1, 0], rot[..., 0, 0])


def _yaw_to_so3(yaw: torch.Tensor) -> torch.Tensor:
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    rot2 = torch.stack([torch.stack([cos, -sin], dim=-1), torch.stack([sin, cos], dim=-1)], dim=-2)
    top = torch.cat([rot2, torch.zeros_like(rot2[..., :1])], dim=-1)
    bottom = torch.tensor([0.0, 0.0, 1.0], dtype=yaw.dtype, device=yaw.device).repeat(rot2.shape[:-2] + (1, 1))
    return torch.cat([top, bottom], dim=-2)


def _third_order_dtd(n: int, lead: tuple, lam: float, dt: float, dtype: torch.dtype, device) -> torch.Tensor:
    """lam / dt^6 * D3^T D3 for the third-order difference operator D3 over n samples."""
    d3 = torch.zeros(*lead, max(n - 3, 0), n, dtype=dtype, device=device)
    rows = torch.arange(max(n - 3, 0), device=device)
    for k, c in enumerate((-1.0, 3.0, -3.0, 1.0)):
        d3[..., rows, rows + k] = c
    return lam / dt**6 * torch.einsum("...ij,...ik->...jk", d3, d3)


def _history_speed(dxy: torch.Tensor, theta: torch.Tensor, dt: float, v_lambda: float, v_ridge: float) -> torch.Tensor:
    """Least-squares speeds at the N + 1 history poses from trapezoidal motion along `theta`, jerk-smoothed."""
    *lead, n, _ = dxy.shape
    a = torch.zeros(*lead, 2 * n, n + 1, dtype=dxy.dtype, device=dxy.device)
    b = (2 / dt * dxy).flatten(start_dim=-2)
    cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
    cos_rows = 2 * torch.arange(n, device=dxy.device)
    cols = torch.arange(n, device=dxy.device)
    a[..., cos_rows, cols] = cos_theta[..., :-1]
    a[..., cos_rows, cols + 1] = cos_theta[..., 1:]
    a[..., cos_rows + 1, cols] = sin_theta[..., :-1]
    a[..., cos_rows + 1, cols + 1] = sin_theta[..., 1:]
    ata = torch.einsum("...ij,...ik->...jk", a, a)
    rhs = torch.einsum("...ij,...i->...j", a, b)
    ridge = v_ridge * torch.eye(n + 1, dtype=dxy.dtype, device=dxy.device).expand(*lead, n + 1, n + 1)
    lhs = ata + _third_order_dtd(n + 1, tuple(lead), v_lambda, dt, dxy.dtype, dxy.device) + ridge
    return torch.cholesky_solve(rhs.unsqueeze(-1), torch.linalg.cholesky(lhs)).squeeze(-1)


def _course_offset(chord_xy: torch.Tensor, yaw: torch.Tensor, chord_dt: float) -> torch.Tensor:
    """Course minus body yaw along a position chord (modulo pi), zeroed past the bound and faded in with speed."""
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    along = cos_yaw * chord_xy[..., 0] + sin_yaw * chord_xy[..., 1]
    across = cos_yaw * chord_xy[..., 1] - sin_yaw * chord_xy[..., 0]
    offset = torch.atan2(across * torch.sign(along), along.abs())
    offset = torch.where(offset.abs() > _MAX_COURSE_OFFSET_RAD, torch.zeros_like(offset), offset)
    speed = torch.linalg.vector_norm(chord_xy, dim=-1) / chord_dt
    fade = (speed - _COURSE_FADE_START_MPS) / (_COURSE_FADE_END_MPS - _COURSE_FADE_START_MPS)
    return offset * fade.clamp(min=0.0, max=1.0)


def estimate_t0_states(hist_xyz: torch.Tensor, hist_rot: torch.Tensor, acfg: ActionConfig) -> dict[str, torch.Tensor]:
    """Speed `v` [m/s] and course `theta` [rad, relative to the t0 rig yaw] at t0 from >= 2 history poses."""
    if hist_xyz.shape[-2] < 2:
        raise ValueError(f"the course-heading action space needs at least 2 history poses, got {hist_xyz.shape[-2]}")
    xy = hist_xyz[..., :2]
    theta = _unwrap_angle(_so3_to_yaw(hist_rot))
    v = _history_speed(xy[..., 1:, :] - xy[..., :-1, :], theta, acfg.dt, acfg.v_lambda, acfg.v_ridge)
    last_yaw = _unwrap_angle(_so3_to_yaw(hist_rot[..., -2:, :, :]))
    course = _course_offset(xy[..., -1, :] - xy[..., -2, :], last_yaw.mean(dim=-1), acfg.dt)
    return {"v": v[..., -1], "theta": course}


def action_to_traj(
    actions: torch.Tensor, hist_xyz: torch.Tensor, hist_rot: torch.Tensor, acfg: ActionConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized (accel, curvature) actions [..., N, 2] -> future xyz [..., N, 3] and rotations [..., N, 3, 3].

    `hist_xyz` [..., T_hist, 3] and `hist_rot` [..., T_hist, 3, 3] are the ego history in the t0 rig frame. The
    unicycle integrates along the course; the rotations carry the course plus a sideslip anchored at t0.
    """
    # The upstream action space stores these as fp32 buffers, so fp64 decoding also uses the fp32-rounded values.
    accel_mean, accel_std, kappa_mean, kappa_std = (
        torch.tensor(value, device=actions.device)
        for value in (acfg.accel_mean, acfg.accel_std, acfg.curvature_mean, acfg.curvature_std)
    )
    accel = actions[..., 0] * accel_std + accel_mean
    kappa = actions[..., 1] * kappa_std + kappa_mean

    t0 = estimate_t0_states(hist_xyz, hist_rot, acfg)
    v0, theta0 = t0["v"], t0["theta"]
    dt = acfg.dt
    velocity = torch.cat([v0.unsqueeze(-1), v0.unsqueeze(-1) + torch.cumsum(accel * dt, dim=-1)], dim=-1)
    theta = torch.cat(
        [
            theta0.unsqueeze(-1),
            theta0.unsqueeze(-1)
            + torch.cumsum(kappa * velocity[..., :-1] * dt, dim=-1)
            + torch.cumsum(kappa * accel * (0.5 * dt**2), dim=-1),
        ],
        dim=-1,
    )
    half_dt = 0.5 * dt
    x = (
        torch.zeros_like(v0).unsqueeze(-1)
        + torch.cumsum(velocity[..., :-1] * torch.cos(theta[..., :-1]) * half_dt, dim=-1)
        + torch.cumsum(velocity[..., 1:] * torch.cos(theta[..., 1:]) * half_dt, dim=-1)
    )
    y = (
        torch.zeros_like(v0).unsqueeze(-1)
        + torch.cumsum(velocity[..., :-1] * torch.sin(theta[..., :-1]) * half_dt, dim=-1)
        + torch.cumsum(velocity[..., 1:] * torch.sin(theta[..., 1:]) * half_dt, dim=-1)
    )
    xyz = torch.zeros(*hist_xyz.shape[:-2], acfg.n_waypoints, 3, dtype=hist_xyz.dtype, device=hist_xyz.device)
    xyz[..., 0] = x
    xyz[..., 1] = y
    xyz[..., 2] = hist_xyz[..., -1:, 2]

    lateral_accel = velocity * torch.gradient(theta, spacing=dt, dim=-1)[0]
    slip_change = acfg.slip_gradient * (lateral_accel - lateral_accel[..., :1])
    yaw = theta - theta0.unsqueeze(-1) + slip_change
    return xyz, _yaw_to_so3(yaw[..., 1:])
