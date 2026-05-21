# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
E2E latency profiler for ED-Pose on TT P150.
Runs the full pipeline twice (cold + warm) in a single process.
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
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import (
    TTEDPoseDecoder,
    MLP,
    inverse_sigmoid,
    gen_sineembed_for_position,
)

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


def load_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in sd.items()}


def prepare_attn_mask2():
    total_q = NUM_GROUP * (NUM_BODY_POINTS + 1)
    group_size = NUM_BODY_POINTS + 1
    kpt_index = [x for x in range(total_q) if x % group_size == 0]
    attn_mask = torch.zeros(1, N_HEADS, total_q, total_q, dtype=torch.bool)
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


def build_prediction_heads(state_dict):
    last = N_DEC_LAYERS - 1
    class_embed = nn.Linear(D_MODEL, NUM_CLASSES)
    ce_sd = {k.replace(f"class_embed.{last}.", ""): v
             for k, v in state_dict.items() if k.startswith(f"class_embed.{last}.")}
    class_embed.load_state_dict(ce_sd)
    class_embed.eval()

    bbox_embed = MLP(D_MODEL, D_MODEL, 4, 3)
    be_sd = {k.replace(f"bbox_embed.{last}.", ""): v
             for k, v in state_dict.items() if k.startswith(f"bbox_embed.{last}.")}
    bbox_embed.load_state_dict(be_sd)
    bbox_embed.eval()

    pose_idx = last - NUM_BOX_DEC_LAYERS
    pose_embed = MLP(D_MODEL, D_MODEL, 2, 3)
    pe_sd = {k.replace(f"pose_embed.{pose_idx}.", ""): v
             for k, v in state_dict.items() if k.startswith(f"pose_embed.{pose_idx}.")}
    pose_embed.load_state_dict(pe_sd)
    pose_embed.eval()

    kpt_index = [x for x in range(NUM_GROUP * (NUM_BODY_POINTS + 1))
                 if x % (NUM_BODY_POINTS + 1) != 0]
    return class_embed, bbox_embed, pose_embed, kpt_index


@torch.no_grad()
def run_pipeline(backbone, encoder, two_stage, decoder, device,
                 image_tensor, mask, attn_mask2, full_sd,
                 class_embed, bbox_embed, pose_embed, kpt_index):
    timings = {}

    # Phase 1: Backbone (CPU)
    t = time.time()
    bb_out = backbone(image_tensor, mask)
    timings["backbone"] = time.time() - t

    # Phase 2: Encoder (ttnn)
    t = time.time()
    src_tt = ttnn.from_torch(
        bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_tt = ttnn.from_torch(
        bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    enc_out_tt = encoder(
        src_tt, pos_tt,
        bb_out["reference_points"], bb_out["spatial_shapes"],
        bb_out["level_start_index"], bb_out["mask_flatten"],
    )
    memory = ttnn.to_torch(enc_out_tt).float()
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    timings["encoder"] = time.time() - t

    # Phase 3: Two-stage (CPU)
    t = time.time()
    query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
    timings["two_stage"] = time.time() - t

    # Phase 4: Decoder (ttnn) — reuse enc_out_tt on device as memory
    t = time.time()
    hs, references = decoder(
        tgt=query_out["tgt"],
        memory_tt=enc_out_tt,
        refpoint_embed=query_out["refpoint_embed"],
        spatial_shapes=bb_out["spatial_shapes"],
        level_start_index=bb_out["level_start_index"],
        valid_ratios=bb_out["valid_ratios"],
        memory_key_padding_mask=bb_out["mask_flatten"],
        self_attn_mask=None,
        self_attn_mask2=attn_mask2,
    )
    ttnn.deallocate(enc_out_tt)
    timings["decoder"] = time.time() - t

    # Phase 5: Heads (CPU)
    t = time.time()
    last_hs = hs[-1]
    last_ref = references[-2]
    stride = NUM_BODY_POINTS + 1
    hs_bbox = last_hs[:, 0::stride, :]
    ref_bbox = last_ref[:, 0::stride, :]
    pred_logits = class_embed(hs_bbox)
    delta = bbox_embed(hs_bbox)
    pred_boxes = (delta + inverse_sigmoid(ref_bbox)).sigmoid()
    scores = pred_logits.sigmoid().max(-1)[0].squeeze(0)
    n_det = (scores > 0.3).sum().item()
    timings["heads"] = time.time() - t
    timings["detections"] = n_det

    return timings


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    full_sd = load_state_dict()
    print(f"Loaded checkpoint: {len(full_sd)} parameters\n")

    print("Building pipeline...")
    t0 = time.time()
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")

    enc_sd = {}
    for k, v in full_sd.items():
        if k.startswith("transformer.encoder.layers."):
            enc_sd[k[len("transformer.encoder."):]] = v.float()
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS
    )

    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)

    decoder = TTEDPoseDecoder(
        device, full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
        N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES, NUM_BODY_POINTS,
        NUM_BOX_DEC_LAYERS, NUM_GROUP,
    )

    class_embed, bbox_embed, pose_embed, kpt_index = build_prediction_heads(full_sd)
    attn_mask2 = prepare_attn_mask2()
    print(f"Pipeline built in {time.time() - t0:.1f}s\n")

    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    N_RUNS = 3
    for run_id in range(N_RUNS):
        label = "COLD" if run_id == 0 else f"WARM-{run_id}"
        print(f"=== Run {run_id} ({label}) ===")
        timings = run_pipeline(
            backbone, encoder, two_stage, decoder, device,
            image_tensor, mask, attn_mask2, full_sd,
            class_embed, bbox_embed, pose_embed, kpt_index,
        )
        total = sum(v for k, v in timings.items() if k != "detections")
        print(f"  Backbone (CPU):   {timings['backbone']*1000:>7.0f}ms")
        print(f"  Encoder  (ttnn):  {timings['encoder']*1000:>7.0f}ms")
        print(f"  Two-stage (CPU):  {timings['two_stage']*1000:>7.0f}ms")
        print(f"  Decoder  (ttnn):  {timings['decoder']*1000:>7.0f}ms")
        print(f"  Heads    (CPU):   {timings['heads']*1000:>7.0f}ms")
        print(f"  Total:            {total*1000:>7.0f}ms")
        print(f"  Detections (>0.3): {timings['detections']}")
        print()

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
