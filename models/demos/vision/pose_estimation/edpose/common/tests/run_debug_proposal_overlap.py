# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compare two-stage proposal selection between fp32 CPU encoder and bf16 ttnn encoder."""

import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoder
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import TwoStageQueryGenerator

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

D_MODEL = 256
D_FFN = 2048
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4
N_ENC_LAYERS = 6
NUM_QUERIES = 900
NUM_CLASSES = 2


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = 800 / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > 1333:
        scale = 1333 / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = normalize(img)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    mask = torch.ones(tensor.shape[1], tensor.shape[2], dtype=torch.bool)
    mask[:new_h, :new_w] = False
    return tensor.unsqueeze(0), mask.unsqueeze(0), torch.tensor([[orig_h, orig_w]])


def load_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in sd.items()}


def build_cpu_encoder(state_dict):
    """Build the reference CPU encoder from ED-Pose checkpoint."""
    from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import CPUMSDeformAttn

    class CPUEncoderLayer(nn.Module):
        def __init__(self, d_model=256, d_ffn=2048, n_levels=5, n_heads=8, n_points=4):
            super().__init__()
            self.self_attn = CPUMSDeformAttn(d_model, n_levels, n_heads, n_points)
            self.norm1 = nn.LayerNorm(d_model)
            self.linear1 = nn.Linear(d_model, d_ffn)
            self.linear2 = nn.Linear(d_ffn, d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
            src2 = self.self_attn(
                src + pos, reference_points, src, spatial_shapes, level_start_index, padding_mask
            )
            src = src + src2
            src = self.norm1(src)
            src2 = self.linear2(F.relu(self.linear1(src)))
            src = src + src2
            src = self.norm2(src)
            return src

    layers = nn.ModuleList()
    for i in range(N_ENC_LAYERS):
        layer = CPUEncoderLayer(D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
        layer_sd = {}
        prefix = f"transformer.encoder.layers.{i}."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                layer_sd[k[len(prefix):]] = v
        layer.load_state_dict(layer_sd, strict=True)
        layer.eval()
        layers.append(layer)
    return layers


def main():
    image_path = "/home/yito/datasets/coco/val2017/000000000139.jpg"

    device = ttnn.open_device(device_id=0)
    full_sd = load_state_dict()

    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")

    # Build ttnn encoder
    enc_sd = {}
    for k, v in full_sd.items():
        if k.startswith("transformer.encoder.layers."):
            enc_sd[k[len("transformer.encoder."):]] = v.float()
    tt_encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS
    )

    # Build CPU encoder
    cpu_encoder_layers = build_cpu_encoder(full_sd)

    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)

    # Preprocess
    tensor, mask, orig_size = preprocess_image(image_path)
    print(f"Image: {tensor.shape}")

    # Backbone
    with torch.no_grad():
        bb_out = backbone(tensor, mask)

    # CPU encoder (fp32)
    print("\n=== CPU Encoder (fp32) ===")
    cpu_memory = bb_out["src_flatten"].clone()
    with torch.no_grad():
        for layer in cpu_encoder_layers:
            cpu_memory = layer(
                cpu_memory, bb_out["pos_flatten"],
                bb_out["reference_points"], bb_out["spatial_shapes"],
                bb_out["level_start_index"], bb_out["mask_flatten"],
            )
    print(f"  Output: mean={cpu_memory.mean():.6f}, std={cpu_memory.std():.6f}")

    # ttnn encoder (bf16)
    print("\n=== ttnn Encoder (bf16) ===")
    src_tt = ttnn.from_torch(
        bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_tt = ttnn.from_torch(
        bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with torch.no_grad():
        enc_out_tt = tt_encoder(
            src_tt, pos_tt, bb_out["reference_points"],
            bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"],
        )
    tt_memory = ttnn.to_torch(enc_out_tt).float()
    ttnn.deallocate(enc_out_tt)
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    print(f"  Output: mean={tt_memory.mean():.6f}, std={tt_memory.std():.6f}")

    # Compare encoder outputs
    diff = (cpu_memory - tt_memory).abs()
    pcc = torch.corrcoef(torch.stack([cpu_memory.flatten(), tt_memory.flatten()]))[0, 1]
    print(f"\n=== Encoder output comparison ===")
    print(f"  PCC:     {pcc:.6f}")
    print(f"  Mean abs diff: {diff.mean():.6f}")
    print(f"  Max abs diff:  {diff.max():.6f}")

    # Two-stage with CPU encoder output
    with torch.no_grad():
        cpu_query = two_stage(cpu_memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
        tt_query = two_stage(tt_memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])

    # Compare proposals
    cpu_ref = cpu_query["refpoint_embed"]
    tt_ref = tt_query["refpoint_embed"]

    cpu_ref_sig = cpu_ref.sigmoid()
    tt_ref_sig = tt_ref.sigmoid()

    print(f"\n=== Two-stage proposal comparison ===")
    for dim, name in enumerate(["cx", "cy", "w", "h"]):
        cpu_vals = cpu_ref_sig[0, :, dim]
        tt_vals = tt_ref_sig[0, :, dim]
        pcc_d = torch.corrcoef(torch.stack([cpu_vals, tt_vals]))[0, 1]
        print(f"  {name}: PCC={pcc_d:.6f}, max_diff={abs(cpu_vals - tt_vals).max():.6f}")

    # Check proposal overlap (how many of the same indices are selected)
    # We need the topk indices to compare
    from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import (
        gen_encoder_output_proposals,
    )

    cpu_out_memory, cpu_out_proposals = gen_encoder_output_proposals(
        cpu_memory, bb_out["mask_flatten"], bb_out["spatial_shapes"]
    )
    cpu_out_memory = two_stage.enc_output_norm(two_stage.enc_output(cpu_out_memory))
    cpu_class = two_stage.enc_out_class_embed(cpu_out_memory)
    cpu_topk_idx = torch.topk(cpu_class.max(-1)[0], 900, dim=1)[1]

    tt_out_memory, tt_out_proposals = gen_encoder_output_proposals(
        tt_memory, bb_out["mask_flatten"], bb_out["spatial_shapes"]
    )
    tt_out_memory = two_stage.enc_output_norm(two_stage.enc_output(tt_out_memory))
    tt_class = two_stage.enc_out_class_embed(tt_out_memory)
    tt_topk_idx = torch.topk(tt_class.max(-1)[0], 900, dim=1)[1]

    cpu_set = set(cpu_topk_idx[0].tolist())
    tt_set = set(tt_topk_idx[0].tolist())
    overlap = len(cpu_set & tt_set)
    print(f"\n  Top-900 proposal overlap: {overlap}/900 ({100*overlap/900:.1f}%)")

    # Check top-100 (used for query expansion)
    cpu_top100_idx = torch.topk(cpu_class.max(-1)[0], 100, dim=1)[1]
    tt_top100_idx = torch.topk(tt_class.max(-1)[0], 100, dim=1)[1]
    cpu_set100 = set(cpu_top100_idx[0].tolist())
    tt_set100 = set(tt_top100_idx[0].tolist())
    overlap100 = len(cpu_set100 & tt_set100)
    print(f"  Top-100 proposal overlap: {overlap100}/100 ({100*overlap100/100:.1f}%)")

    # Check if the h saturation issue is resolved
    print(f"\n=== Proposal h-value analysis ===")
    cpu_h = cpu_ref_sig[0, :, 3]
    tt_h = tt_ref_sig[0, :, 3]
    print(f"  CPU h: mean={cpu_h.mean():.4f}, min={cpu_h.min():.4f}, max={cpu_h.max():.4f}, "
          f"h>=0.99: {(cpu_h >= 0.99).sum()}, h>=0.999: {(cpu_h >= 0.999).sum()}")
    print(f"  TT  h: mean={tt_h.mean():.4f}, min={tt_h.min():.4f}, max={tt_h.max():.4f}, "
          f"h>=0.99: {(tt_h >= 0.99).sum()}, h>=0.999: {(tt_h >= 0.999).sum()}")

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
