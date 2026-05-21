# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 2: Encoder/Decoder layer test — ttnn vs PyTorch reference.
Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_encoder_decoder_test.py
"""

import sys
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoderLayer
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_decoder import TTDeformableDecoderLayer


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


class RefMSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attn_w = F.softmax(
            self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points), dim=-1
        ).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1).float()
            locs = reference_points[:, :, None, :, None, :].float() + offsets / normalizer[None, None, None, :, None, :]
        else:
            locs = (
                reference_points[:, :, None, :, None, :2].float()
                + offsets / self.n_points * reference_points[:, :, None, :, None, 2:].float() * 0.5
            )
        out = ms_deform_attn_core_pytorch(value.float(), input_spatial_shapes, locs.float(), attn_w)
        return self.output_proj(out)


class RefEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = RefMSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index)
        src = self.norm1(src + src2)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + src2)
        return src


class RefDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        self.cross_attn = RefMSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, tgt_query_pos, tgt_reference_points, memory, memory_spatial_shapes,
                memory_level_start_index, self_attn_mask=None):
        q = k = tgt + tgt_query_pos
        tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
        tgt = self.norm2(tgt + tgt2)
        tgt2 = self.cross_attn(
            (tgt + tgt_query_pos).transpose(0, 1), tgt_reference_points.transpose(0, 1),
            memory, memory_spatial_shapes, memory_level_start_index
        ).transpose(0, 1)
        tgt = self.norm1(tgt + tgt2)
        tgt2 = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + tgt2)
        return tgt


def compute_pcc(a, b):
    fa, fb = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([fa, fb]))[0, 1].item()


SPATIAL_SHAPES = torch.tensor([[200, 304], [100, 152], [50, 76], [25, 38], [13, 19]], dtype=torch.long)
LEVEL_START = torch.tensor([0, 60800, 76000, 79800, 80750], dtype=torch.long)
LEN_IN = int((SPATIAL_SHAPES[:, 0] * SPATIAL_SHAPES[:, 1]).sum().item())


def test_encoder_layer(device):
    torch.manual_seed(42)
    N, d_model = 1, 256
    Lq = 960

    ref = RefEncoderLayer().float().eval()
    sd = ref.state_dict()

    src = torch.randn(N, Lq, d_model)
    pos = torch.randn(N, Lq, d_model)
    ref_pts = torch.rand(N, Lq, 5, 2)

    small_shapes = torch.tensor([[10, 16], [8, 12], [4, 6], [2, 3], [2, 2]], dtype=torch.long)
    small_starts = torch.tensor([0, 160, 256, 280, 286], dtype=torch.long)
    small_len = int((small_shapes[:, 0] * small_shapes[:, 1]).sum().item())

    src = torch.randn(N, small_len, d_model)
    pos = torch.randn(N, small_len, d_model)
    ref_pts = torch.rand(N, small_len, 5, 2)

    with torch.no_grad():
        ref_out = ref(src, pos, ref_pts, small_shapes, small_starts)

    tt_prefix = ""
    flat_sd = {}
    for k, v in sd.items():
        flat_sd[k] = v

    tt_layer = TTDeformableEncoderLayer(device, flat_sd, "", d_model=256, d_ffn=1024,
                                         n_levels=5, n_heads=8, n_points=4)

    src_tt = ttnn.from_torch(src.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    pos_tt = ttnn.from_torch(pos.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_layer(src_tt, pos_tt, ref_pts, small_shapes, small_starts)
    t1 = time.time()

    tt_result = ttnn.to_torch(tt_out).float()
    ref_bf16 = ref_out.to(torch.bfloat16).float()
    pcc = compute_pcc(ref_bf16, tt_result)
    elapsed = (t1 - t0) * 1000
    status = "PASS" if pcc > 0.96 else "FAIL"
    print(f"  Encoder Layer (small)          | PCC={pcc:.5f} | {elapsed:8.1f}ms | {status}")
    return pcc > 0.96


def test_decoder_layer(device):
    torch.manual_seed(42)
    N, d_model = 1, 256
    Lq = 900

    ref = RefDecoderLayer().float().eval()
    sd = ref.state_dict()

    small_shapes = torch.tensor([[10, 16], [8, 12], [4, 6], [2, 3], [2, 2]], dtype=torch.long)
    small_starts = torch.tensor([0, 160, 256, 280, 286], dtype=torch.long)
    small_len = int((small_shapes[:, 0] * small_shapes[:, 1]).sum().item())

    tgt = torch.randn(Lq, N, d_model)
    tgt_query_pos = torch.randn(Lq, N, d_model)
    tgt_ref_pts = torch.rand(Lq, N, 5, 4)
    memory = torch.randn(N, small_len, d_model)

    with torch.no_grad():
        ref_out = ref(tgt, tgt_query_pos, tgt_ref_pts, memory, small_shapes, small_starts)

    flat_sd = {k: v for k, v in sd.items()}

    memory_tt = ttnn.from_torch(memory.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    tt_layer = TTDeformableDecoderLayer(device, flat_sd, "", d_model=256, d_ffn=1024,
                                         n_levels=5, n_heads=8, n_points=4, has_self_attn=True)

    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_layer(tgt, tgt_query_pos, tgt_ref_pts, memory_tt,
                          small_shapes, small_starts)
    t1 = time.time()

    ref_bf16 = ref_out.to(torch.bfloat16).float()
    tt_result_bf16 = tt_out.to(torch.bfloat16).float()
    pcc = compute_pcc(ref_bf16, tt_result_bf16)
    elapsed = (t1 - t0) * 1000
    status = "PASS" if pcc > 0.96 else "FAIL"
    print(f"  Decoder Layer (Lq=900)         | PCC={pcc:.5f} | {elapsed:8.1f}ms | {status}")
    return pcc > 0.96


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    results = []

    print("=== Encoder Layer ===")
    results.append(test_encoder_layer(device))

    print("\n=== Decoder Layer ===")
    results.append(test_decoder_layer(device))

    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
