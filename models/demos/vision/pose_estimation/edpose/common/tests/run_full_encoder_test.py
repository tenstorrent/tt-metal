# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Test full 6-layer encoder stack with real ED-Pose checkpoint weights.
Validates that multi-layer composition works correctly.

Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_full_encoder_test.py
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import (
    TTDeformableEncoder,
    TTDeformableEncoderLayer,
)

CHECKPOINT_PATH = "/home/yito/ttwork/ED-Pose/weights/edpose_swinl_5scale_coco.pth"
D_MODEL = 256
D_FFN = 2048
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4
N_LAYERS = 6


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
    def __init__(self, d_model=256, d_ffn=2048, n_levels=5, n_heads=8, n_points=4):
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


class RefEncoder(nn.Module):
    def __init__(self, n_layers=6, d_model=256, d_ffn=2048, n_levels=5, n_heads=8, n_points=4):
        super().__init__()
        self.layers = nn.ModuleList([
            RefEncoderLayer(d_model, d_ffn, n_levels, n_heads, n_points)
            for _ in range(n_layers)
        ])

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index)
        return output


def compute_pcc(a, b):
    fa, fb = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([fa, fb]))[0, 1].item()


def load_encoder_weights(ckpt_path):
    print(f"Loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    enc_sd = {}
    for k, v in cleaned.items():
        if k.startswith("transformer.encoder.layers."):
            new_key = k[len("transformer.encoder."):]
            enc_sd[new_key] = v.float()

    print(f"  Extracted {len(enc_sd)} encoder parameters")
    return enc_sd


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    enc_sd = load_encoder_weights(CHECKPOINT_PATH)

    shapes = torch.tensor([[10, 16], [8, 12], [4, 6], [2, 3], [2, 2]], dtype=torch.long)
    starts = torch.tensor([0, 160, 256, 280, 286], dtype=torch.long)
    length = 290

    torch.manual_seed(42)
    N = 1
    src = torch.randn(N, length, D_MODEL)
    pos = torch.randn(N, length, D_MODEL)
    ref_pts = torch.rand(N, length, N_LEVELS, 2)

    print("=== Reference (PyTorch CPU) ===")
    ref = RefEncoder(N_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS).float().eval()
    ref_sd = ref.state_dict()
    mapping = {k: enc_sd[k] for k in ref_sd if k in enc_sd}
    ref.load_state_dict(mapping, strict=True)
    print(f"  Loaded {len(mapping)} weights")

    t0 = time.time()
    with torch.no_grad():
        ref_out = ref(src, pos, ref_pts, shapes, starts)
    t1 = time.time()
    print(f"  CPU 6-layer encoder: {(t1-t0)*1000:.1f}ms")

    print("\n=== ttnn Encoder (6 layers) ===")
    tt_encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS
    )

    src_tt = ttnn.from_torch(src.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    pos_tt = ttnn.from_torch(pos.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_encoder(src_tt, pos_tt, ref_pts, shapes, starts)
    t1 = time.time()
    elapsed = (t1 - t0) * 1000
    print(f"  ttnn 6-layer encoder: {elapsed:.1f}ms")

    tt_result = ttnn.to_torch(tt_out).float()
    ref_bf16 = ref_out.to(torch.bfloat16).float()
    pcc = compute_pcc(ref_bf16, tt_result)
    status = "PASS" if pcc > 0.95 else "FAIL"
    print(f"\n  Full 6-layer Encoder | PCC={pcc:.5f} | {status}")

    # Also test per-layer drift by running layers incrementally
    print("\n=== Per-layer PCC drift ===")
    ref_intermediate = src.clone()
    tt_intermediate = ttnn.from_torch(src.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    for i in range(N_LAYERS):
        with torch.no_grad():
            ref_intermediate = ref.layers[i](ref_intermediate, pos, ref_pts, shapes, starts)
            tt_intermediate = tt_encoder.layers[i](tt_intermediate, pos_tt, ref_pts, shapes, starts)

        tt_i = ttnn.to_torch(tt_intermediate).float()
        ref_i = ref_intermediate.to(torch.bfloat16).float()
        pcc_i = compute_pcc(ref_i, tt_i)
        print(f"  After layer {i}: PCC={pcc_i:.5f}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
