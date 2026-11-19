# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Action input/output projections and the Euler flow-matching loop around the TT expert.

The Fourier features and the Euler update run on host in fp32: a feature's argument is 2π·freq·value with
frequencies up to `max_freq` (100), hundreds of radians for actions a few sigma out, which a bf16 sin/cos cannot
resolve. Each step uploads the [B, N, 60] features; the device runs the in-proj MLP and LayerNorm, the expert and the
out-proj, and the [B, N, 2] velocity comes back to host.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.alpamayo2_edge.reference.action import FourierEncoder, euler_sample
from models.experimental.alpamayo2_edge.reference.config import ActionConfig
from models.experimental.alpamayo2_edge.tt.common import accurate_compute_config, single_device_ctx
from models.experimental.alpamayo2_edge.tt.expert import ExpertFrame, TtExpert
from models.experimental.alpamayo2_edge.tt.text import AlpamayoCache
from models.tt_dit.encoders.transformer import TransformerRmsNorm
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.utils import tensor

# The trunk's RMSNorm epsilon, as in the reference and upstream `ActionInProj`.
_RMS_EPS = 1e-5


def action_features(x: torch.Tensor, t: torch.Tensor, acfg: ActionConfig) -> torch.Tensor:
    """fp32 Fourier features of actions x [B, N, action_dim] at flow time t [B, 1, 1].

    Returns [B, N, (action_dim + 1) * fourier_feats], laid out as the reference `ActionInProj`'s trunk input: each
    action dim's [sin | cos], then the timestep's.
    """
    batch, n, dims = x.shape
    if dims != acfg.action_dim:
        raise ValueError(f"expected {acfg.action_dim} action dims, got x {tuple(x.shape)}")
    if tuple(t.shape) != (batch, 1, 1):
        raise ValueError(f"expected flow time [{batch}, 1, 1], got {tuple(t.shape)}")
    encoder = FourierEncoder(acfg.fourier_feats, acfg.max_freq)
    x, t = x.float(), t.float()
    action = [encoder(x[:, :, i]) for i in range(dims)]
    timestep = encoder(t[..., -1]).repeat(1, n, 1)
    return torch.cat([*action, timestep], dim=-1)


class TtActionInProj(Module):
    """Fourier features [B, N, 60] -> expert tokens [B, N, out_dim]: the reference trunk MLP and LayerNorm.

    Trunk: Linear + SiLU, then `in_proj_layers` blocks of RMSNorm + Linear, each followed by SiLU but the last.
    """

    def __init__(self, acfg: ActionConfig, out_dim: int, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()
        ctx = single_device_ctx(mesh_device)
        hidden = acfg.in_proj_hidden
        layers = acfg.in_proj_layers
        compute = accurate_compute_config(mesh_device)
        self.mesh_device = mesh_device
        self.in_features = acfg.fourier_feats * (acfg.action_dim + 1)
        self.trunk_in = Linear(
            self.in_features, hidden, activation_fn="silu", mesh_device=mesh_device, compute_kernel_config=compute
        )
        self.norms = ModuleList(TransformerRmsNorm(hidden, eps=_RMS_EPS, ctx=ctx) for _ in range(layers))
        self.linears = ModuleList(
            Linear(
                hidden,
                out_dim if i == layers - 1 else hidden,
                activation_fn=None if i == layers - 1 else "silu",
                mesh_device=mesh_device,
                compute_kernel_config=compute,
            )
            for i in range(layers)
        )
        self.norm = LayerNorm(out_dim, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        super()._prepare_torch_state(state)
        # Checkpoint `encoder.trunk` indices count the SiLUs: Linear 0, SiLU 1, then block i at RMSNorm 2 + 3i,
        # Linear 3 + 3i and (but the last) SiLU 4 + 3i.
        renames = {"encoder.trunk.0.": "trunk_in."}
        for i in range(len(self.linears)):
            renames[f"encoder.trunk.{2 + 3 * i}."] = f"norms.{i}."
            renames[f"encoder.trunk.{3 + 3 * i}."] = f"linears.{i}."
        for key in list(state):
            for old, new in renames.items():
                if key.startswith(old):
                    state[new + key[len(old) :]] = state.pop(key)
                    break

    def forward(self, features: ttnn.Tensor) -> ttnn.Tensor:
        """`features` `[B, N, in_features]` from `action_features` -> `[B, N, out_dim]`."""
        if features.shape[-1] != self.in_features:
            raise ValueError(f"expected {self.in_features} features, got {tuple(features.shape)}")
        x = self.trunk_in.forward(features)
        for norm, linear in zip(self.norms, self.linears):
            x = linear.forward(norm.forward(x))
        return self.norm.forward(x)


class TtActionOutProj(Linear):
    """Expert output [B, N, hidden] -> flow velocity [B, N, action_dim]; checkpoint `expert.action_out_proj.*`."""

    def __init__(self, hidden: int, action_dim: int, *, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__(
            hidden, action_dim, mesh_device=mesh_device, compute_kernel_config=accurate_compute_config(mesh_device)
        )


def action_velocity_tt(
    expert: TtExpert,
    in_proj: TtActionInProj,
    out_proj: TtActionOutProj,
    *,
    cache: AlpamayoCache,
    frame: ExpertFrame,
    x: torch.Tensor,
    t: torch.Tensor,
    acfg: ActionConfig,
) -> torch.Tensor:
    """fp32 flow velocity [B, N, action_dim] at actions x [B, N, action_dim] and time t [B, 1, 1].

    Destructive: the expert overwrites cache rows `[frame.p_pad, frame.p_pad + N)` of every layer; the prefix is left
    as it was.
    """
    batch, n, _ = x.shape
    if (frame.mask.shape[0], frame.mask.shape[2]) != (batch, n):
        raise ValueError(
            f"the frame is for {frame.mask.shape[0]} x {frame.mask.shape[2]} tokens, x is {tuple(x.shape)}"
        )
    features = tensor.from_torch(action_features(x, t, acfg), device=in_proj.mesh_device)
    hidden = expert.forward(in_proj.forward(features), cache=cache, frame=frame)
    return tensor.to_torch(out_proj.forward(hidden)).float()


def sample_actions_tt(
    expert: TtExpert,
    in_proj: TtActionInProj,
    out_proj: TtActionOutProj,
    *,
    cache: AlpamayoCache,
    frame: ExpertFrame,
    noise: torch.Tensor,
    acfg: ActionConfig,
) -> torch.Tensor:
    """Euler-integrate the TT expert flow from `noise` [B, N, action_dim] over `acfg.steps` steps -> fp32 actions.

    The timesteps, dt and update are the reference `euler_sample`'s, in fp32 on host.
    """

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return action_velocity_tt(expert, in_proj, out_proj, cache=cache, frame=frame, x=x, t=t, acfg=acfg)

    return euler_sample(velocity, noise.float(), acfg.steps)
