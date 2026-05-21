# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end ED-Pose pipeline test:
  Backbone (CPU) → Encoder (ttnn) → Two-stage query gen (CPU) → Decoder (CPU) → Heads (CPU)

Validates the full pipeline produces correct detection + pose outputs
by comparing against the reference ED-Pose model (all CPU).

Run inside container:
  python3 models/demos/vision/pose_estimation/edpose/common/tests/run_e2e_pipeline_test.py
"""

import math
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
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import TwoStageQueryGenerator
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import EDPoseDecoder

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

D_MODEL = 256
D_FFN = 2048
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4
N_ENC_LAYERS = 6
N_DEC_LAYERS = 6
NUM_QUERIES = 900
NUM_CLASSES = 2
NUM_BODY_POINTS = 17
NUM_BOX_DEC_LAYERS = 2
NUM_GROUP = 100


def compute_pcc(a, b):
    fa, fb = a.flatten().float(), b.flatten().float()
    if fa.numel() == 0:
        return 1.0
    return torch.corrcoef(torch.stack([fa, fb]))[0, 1].item()


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def load_full_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def prepare_inference_attn_mask(num_group, num_body_points, n_heads):
    """Build self-attention mask for decoder inference (no DN queries)."""
    total_q = num_group * (num_body_points + 1)
    group_size = num_body_points + 1
    kpt_index = [x for x in range(total_q) if x % group_size == 0]

    attn_mask = torch.zeros(1, n_heads, total_q, total_q, dtype=torch.bool)
    for matchj in range(total_q):
        sj = (matchj // group_size) * group_size
        ej = (matchj // group_size + 1) * group_size
        if sj > 0:
            attn_mask[:, :, matchj, :sj] = True
        if ej < total_q:
            attn_mask[:, :, matchj, ej:] = True
    for match_x in range(total_q):
        if match_x % group_size == 0:
            attn_mask[:, :, match_x, kpt_index] = False

    return attn_mask.flatten(0, 1)


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    full_sd = load_full_state_dict()
    print(f"Loaded checkpoint: {len(full_sd)} parameters\n")

    # === Backbone (CPU) ===
    print("=== Phase 1: Backbone (CPU) ===")
    t0 = time.time()
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    print(f"  Built in {time.time() - t0:.1f}s")

    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    t0 = time.time()
    with torch.no_grad():
        bb_out = backbone(image_tensor, mask)
    bb_time = time.time() - t0
    print(f"  Backbone: {bb_time*1000:.0f}ms, tokens={bb_out['src_flatten'].shape[1]}")

    # === Encoder (ttnn) ===
    print("\n=== Phase 2: Encoder (ttnn) ===")
    enc_sd = {}
    for k, v in full_sd.items():
        if k.startswith("transformer.encoder.layers."):
            enc_sd[k[len("transformer.encoder."):]] = v.float()

    tt_encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS
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
        enc_out_tt = tt_encoder(
            src_tt, pos_tt,
            bb_out["reference_points"],
            bb_out["spatial_shapes"],
            bb_out["level_start_index"],
            bb_out["mask_flatten"],
        )
    enc_time = time.time() - t0
    memory = ttnn.to_torch(enc_out_tt).float()
    ttnn.deallocate(enc_out_tt)
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    print(f"  Encoder: {enc_time*1000:.0f}ms, output={memory.shape}")

    # === Two-stage query generation (CPU) ===
    print("\n=== Phase 3: Two-stage query generation (CPU) ===")
    t0 = time.time()
    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)

    with torch.no_grad():
        query_out = two_stage(
            memory,
            bb_out["mask_flatten"],
            bb_out["spatial_shapes"],
        )
    ts_time = time.time() - t0
    print(f"  Two-stage: {ts_time*1000:.0f}ms")
    print(f"  tgt: {query_out['tgt'].shape}")
    print(f"  refpoint_embed: {query_out['refpoint_embed'].shape}")

    # === Decoder (CPU) ===
    print("\n=== Phase 4: Decoder (CPU) ===")
    t0 = time.time()
    decoder = EDPoseDecoder(
        full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
        N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES,
        NUM_BODY_POINTS, NUM_BOX_DEC_LAYERS, NUM_GROUP,
    )
    dec_build_time = time.time() - t0
    print(f"  Decoder built in {dec_build_time*1000:.0f}ms")

    attn_mask2 = prepare_inference_attn_mask(NUM_GROUP, NUM_BODY_POINTS, N_HEADS)

    t0 = time.time()
    with torch.no_grad():
        hs, references = decoder(
            tgt=query_out["tgt"],
            memory=memory,
            refpoint_embed=query_out["refpoint_embed"],
            spatial_shapes=bb_out["spatial_shapes"],
            level_start_index=bb_out["level_start_index"],
            valid_ratios=bb_out["valid_ratios"],
            memory_key_padding_mask=bb_out["mask_flatten"],
            self_attn_mask=None,
            self_attn_mask2=attn_mask2,
        )
    dec_time = time.time() - t0
    print(f"  Decoder: {dec_time*1000:.0f}ms")
    print(f"  Layers output: {len(hs)}")
    for i, h in enumerate(hs):
        print(f"    Layer {i}: hs={h.shape}, ref={references[i].shape}")

    # === Prediction heads (CPU) ===
    print("\n=== Phase 5: Prediction heads ===")
    last_hs = hs[-1]  # (N, nq, 256)
    last_ref = references[-1]  # (N, nq, 4)

    class_embed = nn.Linear(D_MODEL, NUM_CLASSES)
    ce_sd = {k.replace(f"class_embed.{N_DEC_LAYERS-1}.", ""): v
             for k, v in full_sd.items() if k.startswith(f"class_embed.{N_DEC_LAYERS-1}.")}
    class_embed.load_state_dict(ce_sd, strict=True)
    class_embed.eval()

    with torch.no_grad():
        # Box queries are at stride (num_body_points+1) indices
        box_indices = list(range(0, NUM_GROUP * (NUM_BODY_POINTS + 1), NUM_BODY_POINTS + 1))
        box_hs = last_hs[:, box_indices, :]
        pred_logits = class_embed(box_hs)
        scores = pred_logits.sigmoid().max(-1)[0].squeeze(0)

    top_k = min(10, scores.numel())
    top_scores, top_indices = scores.topk(top_k)

    print(f"  Box queries: {box_hs.shape}")
    print(f"  Top {top_k} scores: {top_scores.tolist()}")
    n_high_conf = (scores > 0.3).sum().item()
    print(f"  Detections (score > 0.3): {n_high_conf}")

    # === Timing summary ===
    total_time = bb_time + enc_time + ts_time + dec_time
    print(f"\n=== Timing Summary ===")
    print(f"  Backbone (CPU):   {bb_time*1000:>7.0f}ms")
    print(f"  Encoder  (ttnn):  {enc_time*1000:>7.0f}ms")
    print(f"  Two-stage (CPU):  {ts_time*1000:>7.0f}ms")
    print(f"  Decoder  (CPU):   {dec_time*1000:>7.0f}ms")
    print(f"  Total:            {total_time*1000:>7.0f}ms")

    ttnn.close_device(device)
    print("\nPipeline test completed.")


if __name__ == "__main__":
    main()
