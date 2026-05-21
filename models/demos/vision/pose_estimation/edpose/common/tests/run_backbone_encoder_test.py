# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test: Backbone (CPU) → Encoder (ttnn device).
Validates the full pipeline from image to encoder output.

Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_backbone_encoder_test.py
"""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoder

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

D_MODEL = 256
D_FFN = 2048
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4
N_LAYERS = 6


def compute_pcc(a, b):
    fa, fb = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([fa, fb]))[0, 1].item()


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

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
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

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask)
        return output


def load_encoder_weights(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    enc_sd = {}
    for k, v in cleaned.items():
        if k.startswith("transformer.encoder.layers."):
            new_key = k[len("transformer.encoder."):]
            enc_sd[new_key] = v.float()
    return enc_sd


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    # Build backbone
    print("=== Building backbone (CPU) ===")
    t0 = time.time()
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    print(f"  Backbone built in {time.time() - t0:.1f}s")

    # Create synthetic input
    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    # Run backbone
    print("\n=== Running backbone ===")
    t0 = time.time()
    with torch.no_grad():
        bb_out = backbone(image_tensor, mask)
    bb_time = time.time() - t0
    print(f"  Backbone: {bb_time*1000:.0f}ms")
    print(f"  src: {bb_out['src_flatten'].shape}")
    print(f"  spatial_shapes: {bb_out['spatial_shapes'].tolist()}")
    total_tokens = bb_out["src_flatten"].shape[1]
    print(f"  Total tokens: {total_tokens}")

    # Load encoder weights
    enc_sd = load_encoder_weights(CHECKPOINT_PATH)
    print(f"  Loaded {len(enc_sd)} encoder weight tensors")

    # Build reference encoder on CPU
    print("\n=== Reference Encoder (CPU) ===")
    ref_encoder = RefEncoder(N_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS).float().eval()
    ref_sd = ref_encoder.state_dict()
    mapping = {k: enc_sd[k] for k in ref_sd if k in enc_sd}
    ref_encoder.load_state_dict(mapping, strict=True)

    t0 = time.time()
    with torch.no_grad():
        ref_out = ref_encoder(
            bb_out["src_flatten"].float(),
            bb_out["pos_flatten"].float(),
            bb_out["reference_points"].float(),
            bb_out["spatial_shapes"],
            bb_out["level_start_index"],
            bb_out["mask_flatten"],
        )
    ref_time = time.time() - t0
    print(f"  CPU encoder: {ref_time*1000:.0f}ms")

    # Build ttnn encoder
    print("\n=== ttnn Encoder ===")
    tt_encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS
    )

    src_tt = ttnn.from_torch(
        bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_tt = ttnn.from_torch(
        bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_encoder(
            src_tt, pos_tt,
            bb_out["reference_points"],
            bb_out["spatial_shapes"],
            bb_out["level_start_index"],
            bb_out["mask_flatten"],
        )
    tt_time = time.time() - t0
    print(f"  ttnn encoder: {tt_time*1000:.0f}ms")

    tt_result = ttnn.to_torch(tt_out).float()
    ref_bf16 = ref_out.to(torch.bfloat16).float()
    pcc = compute_pcc(ref_bf16, tt_result)
    status = "PASS" if pcc > 0.95 else "FAIL"

    print(f"\n=== Results ===")
    print(f"  Backbone + Encoder Pipeline PCC: {pcc:.5f} | {status}")
    print(f"  Backbone time: {bb_time*1000:.0f}ms")
    print(f"  Encoder time (CPU ref): {ref_time*1000:.0f}ms")
    print(f"  Encoder time (ttnn): {tt_time*1000:.0f}ms")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
