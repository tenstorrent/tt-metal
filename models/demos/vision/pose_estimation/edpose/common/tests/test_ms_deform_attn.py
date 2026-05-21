# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1: MSDeformAttn ttnn implementation test.

Validates that TTMSDeformAttn produces correct results against a pure PyTorch
reference for ED-Pose's encoder and decoder tensor shapes.

ED-Pose config: d_model=256, n_heads=8, n_levels=5, n_points=4
  Feature map sizes: 200x304, 100x152, 50x76, 25x38, 13x19
  Total flattened: 80997 tokens
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_ms_deform_attn import TTMSDeformAttn
from tests.ttnn.utils_for_testing import assert_with_pcc


SPATIAL_SHAPES = torch.tensor([[200, 304], [100, 152], [50, 76], [25, 38], [13, 19]], dtype=torch.long)
LEVEL_START = torch.tensor([0, 60800, 76000, 79800, 80750], dtype=torch.long)
LEN_IN = int((SPATIAL_SHAPES[:, 0] * SPATIAL_SHAPES[:, 1]).sum().item())

D_MODEL = 256
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, int(H_), int(W_))
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(
        N_, M_ * D_, Lq_
    )
    return output.transpose(1, 2).contiguous()


class _RefMSDeformAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = N_HEADS
        self.n_levels = N_LEVELS
        self.n_points = N_POINTS
        self.d_model = D_MODEL
        self.sampling_offsets = nn.Linear(D_MODEL, N_HEADS * N_LEVELS * N_POINTS * 2)
        self.attention_weights = nn.Linear(D_MODEL, N_HEADS * N_LEVELS * N_POINTS)
        self.value_proj = nn.Linear(D_MODEL, D_MODEL)
        self.output_proj = nn.Linear(D_MODEL, D_MODEL)

    def forward(self, query, reference_points, input_flatten):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attn_w = F.softmax(
            self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points), dim=-1
        ).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            normalizer = torch.stack([SPATIAL_SHAPES[..., 1], SPATIAL_SHAPES[..., 0]], -1).float()
            locs = reference_points[:, :, None, :, None, :].float() + offsets / normalizer[None, None, None, :, None, :]
        else:
            locs = (
                reference_points[:, :, None, :, None, :2].float()
                + offsets / self.n_points * reference_points[:, :, None, :, None, 2:].float() * 0.5
            )

        out = ms_deform_attn_core_pytorch(value.float(), SPATIAL_SHAPES, locs.float(), attn_w)
        return self.output_proj(out)


def _run(device, Lq, ref_dim, pcc_threshold=0.97):
    torch.manual_seed(42)
    N = 1
    ref = _RefMSDeformAttn().float().eval()
    sd = {k: v for k, v in ref.state_dict().items()}

    query = torch.randn(N, Lq, D_MODEL)
    input_flat = torch.randn(N, LEN_IN, D_MODEL)
    ref_pts = torch.rand(N, Lq, N_LEVELS, ref_dim)

    with torch.no_grad():
        ref_out = ref(query, ref_pts, input_flat)

    tt_attn = TTMSDeformAttn(device, sd, "", D_MODEL, N_HEADS, N_LEVELS, N_POINTS)
    q_tt = ttnn.from_torch(query.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    i_tt = ttnn.from_torch(input_flat.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    with torch.no_grad():
        tt_out = tt_attn(q_tt, ref_pts, i_tt, SPATIAL_SHAPES, LEVEL_START)

    tt_result = ttnn.to_torch(tt_out)
    assert_with_pcc(ref_out.to(torch.bfloat16), tt_result, pcc=pcc_threshold)


def test_decoder_box_ref4(device):
    _run(device, 900, 4)


def test_decoder_pose_ref4(device):
    _run(device, 1800, 4)


def test_decoder_ref2(device):
    _run(device, 900, 2)


def test_encoder_ref2(device):
    _run(device, 80997, 2)
