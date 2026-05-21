# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test: validates ttnn encoder layer against actual ED-Pose checkpoint weights.

Loads the Swin-L 5scale checkpoint, extracts encoder layer 0 weights,
creates synthetic inputs matching ED-Pose encoder dimensions, and compares
ttnn output against PyTorch reference with the same weights.

Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_encoder_integration_test.py
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoderLayer
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_decoder import TTDeformableDecoderLayer

CHECKPOINT_PATH = "/home/yito/ttwork/ED-Pose/weights/edpose_swinl_5scale_coco.pth"

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

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes, level_start_index, key_padding_mask)
        src = self.norm1(src + src2)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + src2)
        return src


def load_encoder_layer_weights(ckpt_path, layer_idx=0):
    """Load weights for encoder layer from ED-Pose checkpoint."""
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    enc_prefix = f"transformer.encoder.layers.{layer_idx}"
    layer_sd = {}
    for k, v in cleaned.items():
        if k.startswith(enc_prefix):
            new_key = k[len(enc_prefix) + 1:]
            layer_sd[new_key] = v.float()

    print(f"  Extracted {len(layer_sd)} parameters for encoder layer {layer_idx}")
    return layer_sd


def load_decoder_layer_weights(ckpt_path, layer_idx=0):
    """Load weights for decoder layer from ED-Pose checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
    cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}

    dec_prefix = f"transformer.decoder.layers.{layer_idx}"
    layer_sd = {}
    for k, v in cleaned.items():
        if k.startswith(dec_prefix):
            new_key = k[len(dec_prefix) + 1:]
            layer_sd[new_key] = v.float()

    print(f"  Extracted {len(layer_sd)} parameters for decoder layer {layer_idx}")
    return layer_sd


def compute_pcc(a, b):
    fa, fb = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([fa, fb]))[0, 1].item()


def test_encoder_layer_real_weights(device, ckpt_path, layer_idx=0, use_small=True):
    torch.manual_seed(0)
    layer_sd = load_encoder_layer_weights(ckpt_path, layer_idx)

    ref = RefEncoderLayer(D_MODEL, 2048, N_LEVELS, N_HEADS, N_POINTS).float().eval()
    ref_sd = ref.state_dict()
    mapping = {}
    for ref_key in ref_sd:
        if ref_key in layer_sd:
            mapping[ref_key] = layer_sd[ref_key]
        else:
            print(f"  WARNING: {ref_key} not found in checkpoint")
    ref.load_state_dict(mapping, strict=True)
    print(f"  Loaded {len(mapping)} weights into reference model")

    if use_small:
        shapes = torch.tensor([[10, 16], [8, 12], [4, 6], [2, 3], [2, 2]], dtype=torch.long)
        starts = torch.tensor([0, 160, 256, 280, 286], dtype=torch.long)
        length = 290
    else:
        shapes = SPATIAL_SHAPES
        starts = LEVEL_START
        length = LEN_IN

    N = 1
    src = torch.randn(N, length, D_MODEL)
    pos = torch.randn(N, length, D_MODEL)
    ref_pts = torch.rand(N, length, N_LEVELS, 2)

    with torch.no_grad():
        ref_out = ref(src, pos, ref_pts, shapes, starts)

    tt_layer = TTDeformableEncoderLayer(device, layer_sd, "", D_MODEL, 2048, N_LEVELS, N_HEADS, N_POINTS)
    src_tt = ttnn.from_torch(src.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    pos_tt = ttnn.from_torch(pos.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_layer(src_tt, pos_tt, ref_pts, shapes, starts)
    t1 = time.time()

    tt_result = ttnn.to_torch(tt_out).float()
    ref_bf16 = ref_out.to(torch.bfloat16).float()
    pcc = compute_pcc(ref_bf16, tt_result)
    elapsed = (t1 - t0) * 1000
    status = "PASS" if pcc > 0.96 else "FAIL"
    size_label = "small" if use_small else "full"
    print(f"  Encoder L{layer_idx} ({size_label:5s}) real wt  | PCC={pcc:.5f} | {elapsed:8.1f}ms | {status}")
    return pcc > 0.96


def test_decoder_layer_real_weights(device, ckpt_path, layer_idx=0):
    torch.manual_seed(0)
    layer_sd = load_decoder_layer_weights(ckpt_path, layer_idx)

    has_sa = "self_attn.in_proj_weight" in layer_sd

    from models.demos.vision.pose_estimation.edpose.common.tests.run_encoder_decoder_test import RefDecoderLayer
    ref = RefDecoderLayer(D_MODEL, 2048, N_LEVELS, N_HEADS, N_POINTS).float().eval()
    ref_sd = ref.state_dict()
    mapping = {}
    for ref_key in ref_sd:
        if ref_key in layer_sd:
            mapping[ref_key] = layer_sd[ref_key]
        elif ref_key.startswith("self_attn.") and not has_sa:
            continue
        else:
            print(f"  WARNING: {ref_key} not found in checkpoint")
    ref.load_state_dict(mapping, strict=False)
    print(f"  Loaded weights for decoder layer {layer_idx} (self_attn={'yes' if has_sa else 'no'})")

    shapes = torch.tensor([[10, 16], [8, 12], [4, 6], [2, 3], [2, 2]], dtype=torch.long)
    starts = torch.tensor([0, 160, 256, 280, 286], dtype=torch.long)
    small_len = 290

    N = 1
    Lq = 900
    tgt = torch.randn(Lq, N, D_MODEL)
    tgt_pos = torch.randn(Lq, N, D_MODEL)
    tgt_ref = torch.rand(Lq, N, N_LEVELS, 4)
    memory = torch.randn(N, small_len, D_MODEL)

    with torch.no_grad():
        ref_out = ref(tgt, tgt_pos, tgt_ref, memory, shapes, starts)

    memory_tt = ttnn.from_torch(memory.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    tt_layer = TTDeformableDecoderLayer(device, layer_sd, "", D_MODEL, 2048, N_LEVELS, N_HEADS, N_POINTS,
                                         has_self_attn=has_sa)

    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_layer(tgt, tgt_pos, tgt_ref, memory_tt, shapes, starts)
    t1 = time.time()

    ref_bf16 = ref_out.to(torch.bfloat16).float()
    tt_result = tt_out.to(torch.bfloat16).float()
    pcc = compute_pcc(ref_bf16, tt_result)
    elapsed = (t1 - t0) * 1000
    sa_label = "w/SA" if has_sa else "no SA"
    status = "PASS" if pcc > 0.96 else "FAIL"
    print(f"  Decoder L{layer_idx} ({sa_label:5s}) real wt | PCC={pcc:.5f} | {elapsed:8.1f}ms | {status}")
    return pcc > 0.96


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    results = []

    print("=== Encoder Layers (real checkpoint weights) ===")
    results.append(test_encoder_layer_real_weights(device, CHECKPOINT_PATH, 0))
    results.append(test_encoder_layer_real_weights(device, CHECKPOINT_PATH, 5))

    print("\n=== Decoder Layers (real checkpoint weights) ===")
    results.append(test_decoder_layer_real_weights(device, CHECKPOINT_PATH, 0))
    results.append(test_decoder_layer_real_weights(device, CHECKPOINT_PATH, 3))

    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
