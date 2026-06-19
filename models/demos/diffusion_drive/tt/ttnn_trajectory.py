# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TTNN trajectory-head denoiser — Stage 3.5.

Drop-in TTNN replacements for the weight-bearing modules inside
``TrajectoryHead`` / ``CustomTransformerDecoderLayer`` so the DDIM denoiser's
compute runs on-device:

  * ``plan_anchor_encoder``  Linear+ReLU+LN+Linear        → TtnnSequentialMLP
  * ``time_mlp``             SinusoidalPosEmb+Linear+Mish+Linear → TtnnTimeMlp
  * per decoder layer:
      - ``cross_bev_attention``  GridSample deformable    → TtnnGridSampleCrossBEVAttention
      - ``cross_agent_attention``/``cross_ego_attention`` nn.MHA → TtnnMHAModule
      - ``ffn``                  Linear+ReLU+Linear        → TtnnSequentialMLP
      - ``norm1/2/3``            LayerNorm                 → TtnnLayerNormModule
      - ``time_modulation``      ModulationLayer (FiLM)    → TtnnModulation
      - ``task_decoder``         cls/reg MLP heads         → TtnnTaskDecoder

Each drop-in keeps the call signature of the module it replaces, so
``TrajectoryHead._forward_test`` is untouched.  The thin scalar glue that stays
on host (DDIM ``scheduler.step``, ``gen_sineembed_for_position``, ``_norm_odo``/
``_denorm_odo``, ``clamp``, ``argmax``/``gather``) is elementwise/indexing-only
and is the Stage-3.7 single-graph consolidation target.

All math runs in bfloat16; verified PCC ≥ 0.99 vs the PyTorch reference.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import ttnn
from models.demos.diffusion_drive.tt.ttnn_grid_sample_attention import TtnnGridSampleCrossBEVAttention
from models.demos.diffusion_drive.tt.ttnn_perception import _prep_layernorm, _prep_linear, _TtnnMHA


class TtnnSequentialMLP(nn.Module):
    """Run an nn.Sequential of {Linear, ReLU, LayerNorm, Mish, Dropout} on TTNN.

    torch in → torch out.  Covers plan_anchor_encoder, ffn, and the cls/reg
    branches of the task decoder.
    """

    def __init__(self, seq, device) -> None:
        super().__init__()
        self._d = device
        self._ops = []
        for m in seq:
            if isinstance(m, nn.Linear):
                w, b = _prep_linear(m, device)
                self._ops.append(("linear", w, b))
            elif isinstance(m, nn.ReLU):
                self._ops.append(("relu",))
            elif isinstance(m, nn.LayerNorm):
                g, be, eps = _prep_layernorm(m, device)
                self._ops.append(("ln", g, be, eps))
            elif isinstance(m, nn.Mish):
                self._ops.append(("mish",))
            elif isinstance(m, (nn.Dropout, nn.Identity)):
                continue
            else:
                raise ValueError(f"TtnnSequentialMLP: unsupported layer {type(m)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        for op in self._ops:
            if op[0] == "linear":
                xt = ttnn.linear(xt, op[1], bias=op[2])
            elif op[0] == "relu":
                xt = ttnn.relu(xt)
            elif op[0] == "ln":
                xt = ttnn.layer_norm(xt, weight=op[1], bias=op[2], epsilon=op[3])
            elif op[0] == "mish":
                xt = ttnn.mish(xt)
        return ttnn.to_torch(xt).float()


class TtnnTimeMlp(nn.Module):
    """time_mlp drop-in: SinusoidalPosEmb (host, scalar) + Linear/Mish/Linear (TTNN)."""

    def __init__(self, time_mlp: nn.Sequential, device) -> None:
        super().__init__()
        self._sin = time_mlp[0]  # SinusoidalPosEmb — scalar trig, stays host
        self._rest = TtnnSequentialMLP(time_mlp[1:], device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._rest(self._sin(x))


class TtnnLayerNormModule(nn.Module):
    """nn.LayerNorm drop-in on TTNN.  torch in → torch out."""

    def __init__(self, ln: nn.LayerNorm, device) -> None:
        super().__init__()
        self._d = device
        self._g, self._b, self._eps = _prep_layernorm(ln, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        y = ttnn.layer_norm(xt, weight=self._g, bias=self._b, epsilon=self._eps)
        return ttnn.to_torch(y).float()


class TtnnMHAModule(nn.Module):
    """nn.MultiheadAttention drop-in (key==value).  Returns (out, None) tuple."""

    def __init__(self, mha: nn.MultiheadAttention, device) -> None:
        super().__init__()
        self._d = device
        self._mha = _TtnnMHA(mha, device)

    def forward(self, query, key, value, **kwargs):
        B, Sq, _ = query.shape
        Skv = key.shape[1]
        q = ttnn.from_torch(query.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        kv = ttnn.from_torch(key.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        out = self._mha(q, kv, B, Sq, Skv)
        return ttnn.to_torch(out).float(), None


class TtnnModulation(nn.Module):
    """ModulationLayer (FiLM) drop-in: traj * (1 + scale) + shift, on TTNN.

    The single Linear(D → 2D) is pre-split into scale/shift halves so no
    device-side chunk is needed.
    """

    def __init__(self, mod, device) -> None:
        super().__init__()
        self._d = device
        lin = mod.scale_shift_mlp[1]  # Sequential(Mish, Linear(D, 2D))
        E = lin.out_features // 2
        W = lin.weight.detach()
        b = lin.bias.detach()

        def to_w(Wp, bp):
            return (
                ttnn.from_torch(Wp.T.contiguous().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(bp.reshape(1, -1).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device),
            )

        self._ws, self._wsb = to_w(W[:E], b[:E])  # scale
        self._wsh, self._wshb = to_w(W[E:], b[E:])  # shift

    def forward(self, traj_feature: torch.Tensor, time_embed: torch.Tensor) -> torch.Tensor:
        tf = ttnn.from_torch(traj_feature.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        te = ttnn.from_torch(time_embed.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self._d)
        m = ttnn.mish(te)
        scale = ttnn.linear(m, self._ws, bias=self._wsb)  # (B,1,E)
        shift = ttnn.linear(m, self._wsh, bias=self._wshb)  # (B,1,E)
        out = ttnn.add(ttnn.mul(tf, ttnn.add(scale, 1.0)), shift)  # broadcast over query dim
        return ttnn.to_torch(out).float()


class TtnnTaskDecoder(nn.Module):
    """DiffMotionPlanningRefinementModule drop-in: cls/reg MLP heads on TTNN."""

    def __init__(self, mod, device) -> None:
        super().__init__()
        self._cls = TtnnSequentialMLP(mod.plan_cls_branch, device)
        self._reg = TtnnSequentialMLP(mod.plan_reg_branch, device)
        self._ts = mod.ego_fut_ts
        self._mode = mod.ego_fut_mode

    def forward(self, traj_feature: torch.Tensor):
        bs = traj_feature.shape[0]
        plan_cls = self._cls(traj_feature).squeeze(-1)  # (B, K)
        traj_delta = self._reg(traj_feature)  # (B, K, T*3)
        plan_reg = traj_delta.reshape(bs, self._mode, self._ts, 3)
        return plan_reg, plan_cls


class TtnnCrossBevModule(nn.Module):
    """nn.Module wrapper around TtnnGridSampleCrossBEVAttention.

    Needed because the callable is not itself an nn.Module, and PyTorch rejects
    assigning a non-Module over a registered submodule (``cross_bev_attention``).
    """

    def __init__(self, ref, device) -> None:
        super().__init__()
        self._impl = TtnnGridSampleCrossBEVAttention(ref, device)

    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        return self._impl(queries, traj_points, bev_feature, spatial_shape)


def install_ttnn_agent_head(agent_head, device) -> None:
    """Monkeypatch an AgentHead's two MLPs with TTNN drop-ins.

    ``_mlp_states`` (Linear+ReLU+Linear) and ``_mlp_label`` (Linear) run on
    device.  The per-index tanh scaling in ``AgentHead.forward`` (POINT·32,
    HEADING·π) is index-selection glue on a tiny tensor and stays on host.
    The agent head does not affect trajectory/scores; this just removes the
    last weight-bearing PyTorch fallback in the model.
    """
    agent_head._mlp_states = TtnnSequentialMLP(agent_head._mlp_states, device)
    agent_head._mlp_label = TtnnSequentialMLP(agent_head._mlp_label, device)


def install_ttnn_trajectory_head(trajectory_head, device) -> None:
    """Monkeypatch a TrajectoryHead's weight-bearing submodules with TTNN drop-ins."""
    th = trajectory_head
    th.plan_anchor_encoder = TtnnSequentialMLP(th.plan_anchor_encoder, device)
    th.time_mlp = TtnnTimeMlp(th.time_mlp, device)
    for layer in th.diff_decoder.layers:
        layer.cross_bev_attention = TtnnCrossBevModule(layer.cross_bev_attention, device)
        layer.cross_agent_attention = TtnnMHAModule(layer.cross_agent_attention, device)
        layer.cross_ego_attention = TtnnMHAModule(layer.cross_ego_attention, device)
        layer.ffn = TtnnSequentialMLP(layer.ffn, device)
        layer.norm1 = TtnnLayerNormModule(layer.norm1, device)
        layer.norm2 = TtnnLayerNormModule(layer.norm2, device)
        layer.norm3 = TtnnLayerNormModule(layer.norm3, device)
        layer.time_modulation = TtnnModulation(layer.time_modulation, device)
        layer.task_decoder = TtnnTaskDecoder(layer.task_decoder, device)
