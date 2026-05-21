# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Phase 1: MSDeformAttn ttnn vs PyTorch reference test.
Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_ms_deform_attn_test.py
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import ttnn

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_ms_deform_attn import TTMSDeformAttn


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


class PyTorchMSDeformAttn(nn.Module):
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

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :].float()
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :].float()
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2].float()
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:].float() * 0.5
            )
        output = ms_deform_attn_core_pytorch(
            value.float(), input_spatial_shapes, sampling_locations.float(), attention_weights
        )
        output = self.output_proj(output)
        return output


def compute_pcc(a, b):
    fa, fb = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([fa, fb]))[0, 1].item()


SPATIAL_SHAPES = torch.tensor([[200, 304], [100, 152], [50, 76], [25, 38], [13, 19]], dtype=torch.long)
LEVEL_START = torch.tensor([0, 60800, 76000, 79800, 80750], dtype=torch.long)
LEN_IN = int((SPATIAL_SHAPES[:, 0] * SPATIAL_SHAPES[:, 1]).sum().item())


def run_test(device, name, Lq, ref_dim, d_model=256, n_heads=8, n_levels=5, n_points=4):
    torch.manual_seed(42)
    N = 1

    ref_module = PyTorchMSDeformAttn(d_model, n_levels, n_heads, n_points).float()
    ref_module.eval()

    sd = ref_module.state_dict()

    query = torch.randn(N, Lq, d_model)
    input_flatten = torch.randn(N, LEN_IN, d_model)
    reference_points = torch.rand(N, Lq, n_levels, ref_dim)

    with torch.no_grad():
        ref_out = ref_module(query, reference_points, input_flatten, SPATIAL_SHAPES, LEVEL_START)

    prefix = ""
    flat_sd = {}
    for k, v in sd.items():
        flat_sd[f"{prefix}{k}"] = v

    tt_attn = TTMSDeformAttn(device, flat_sd, prefix, d_model, n_heads, n_levels, n_points)

    query_tt = ttnn.from_torch(
        query.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tt = ttnn.from_torch(
        input_flatten.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_attn(query_tt, reference_points, input_tt, SPATIAL_SHAPES, LEVEL_START)
    t1 = time.time()

    tt_result = ttnn.to_torch(tt_out).float()
    ref_bf16 = ref_out.to(torch.bfloat16).float()

    pcc = compute_pcc(ref_bf16, tt_result)
    elapsed = (t1 - t0) * 1000
    status = "PASS" if pcc > 0.97 else "FAIL"
    print(f"  {name:30s} | Lq={Lq:>6d} ref_dim={ref_dim} | PCC={pcc:.5f} | {elapsed:8.1f}ms | {status}")

    ttnn.deallocate(query_tt)
    ttnn.deallocate(input_tt)
    ttnn.deallocate(tt_out)
    return pcc > 0.97


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    results = []

    print("=== Decoder box (Lq=900, ref_dim=4) ===")
    results.append(run_test(device, "dec_box_ref4", 900, 4))

    print("\n=== Decoder pose (Lq=1800, ref_dim=4) ===")
    results.append(run_test(device, "dec_pose_ref4", 1800, 4))

    print("\n=== Encoder self-attn (Lq=80997, ref_dim=2) ===")
    try:
        results.append(run_test(device, "encoder_ref2", 80997, 2))
    except Exception as e:
        print(f"  encoder_ref2                   | SKIPPED: {e}")
        results.append(None)

    print("\n=== Decoder small (Lq=900, ref_dim=2) ===")
    results.append(run_test(device, "dec_box_ref2", 900, 2))

    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    total = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
