# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test: full pipeline x2 to isolate hang."""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))
if TT_METAL_HOME not in sys.path:
    sys.path.insert(0, TT_METAL_HOME)
os.environ.setdefault("EDPOSE_ROOT", os.path.expanduser("~/ttwork/ED-Pose"))

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_swin_backbone import TTSwinLBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoder
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import TwoStageQueryGenerator
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import (
    TTEDPoseDecoder, MLP,
)

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")
IMAGE_PATH = "/home/yito/datasets/coco/val2017/000000000139.jpg"

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


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = 800 / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > 1333:
        scale = 1333 / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = normalize(img)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    mask = torch.ones(tensor.shape[1], tensor.shape[2], dtype=torch.bool)
    mask[:new_h, :new_w] = False
    return tensor.unsqueeze(0), mask.unsqueeze(0)


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
        ej = sj + group_size
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
    full_sd = load_state_dict()

    print("Building full pipeline...")
    t0 = time.time()
    backbone = TTSwinLBackbone(device, CHECKPOINT_PATH)
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    decoder = TTEDPoseDecoder(
        device, full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
        N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES, NUM_BODY_POINTS,
        NUM_BOX_DEC_LAYERS, NUM_GROUP)
    attn_mask2 = prepare_attn_mask2()
    print(f"Built in {time.time() - t0:.1f}s")

    tensor, mask = preprocess_image(IMAGE_PATH)
    print(f"Image padded: {tensor.shape[2]}x{tensor.shape[3]}\n")

    for i in range(2):
        print(f"=== Run {i+1}/2 ===")

        t = time.time()
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        print(f"  backbone: {(time.time()-t)*1000:.0f}ms")

        t = time.time()
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"  src/pos transferred")
        with torch.no_grad():
            enc_out_tt = encoder(src_tt, pos_tt, bb_out["reference_points"],
                bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"])
        memory = ttnn.to_torch(enc_out_tt).float()
        ttnn.deallocate(src_tt)
        ttnn.deallocate(pos_tt)
        print(f"  encoder: {(time.time()-t)*1000:.0f}ms")

        t = time.time()
        with torch.no_grad():
            query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
        print(f"  two_stage: {(time.time()-t)*1000:.0f}ms")

        t = time.time()
        print(f"  starting decoder...")
        with torch.no_grad():
            hs, references = decoder(
                tgt=query_out["tgt"], memory_tt=enc_out_tt,
                refpoint_embed=query_out["refpoint_embed"],
                spatial_shapes=bb_out["spatial_shapes"],
                level_start_index=bb_out["level_start_index"],
                valid_ratios=bb_out["valid_ratios"],
                memory_key_padding_mask=bb_out["mask_flatten"],
                self_attn_mask=None, self_attn_mask2=attn_mask2)
        ttnn.deallocate(enc_out_tt)
        print(f"  decoder: {(time.time()-t)*1000:.0f}ms")

        last_hs = hs[-1]
        stride = NUM_BODY_POINTS + 1
        hs_bbox = last_hs[:, 0::stride, :]
        print(f"  hs_bbox shape: {hs_bbox.shape}")
        print(f"  Run {i+1} complete\n")

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
