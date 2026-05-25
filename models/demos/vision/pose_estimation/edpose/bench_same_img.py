# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ED-Pose: 5 runs on the same image, report warm average."""

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

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_swin_backbone import TTSwinBackbone, TTSwinLBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoder
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import TwoStageQueryGenerator
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import (
    TTEDPoseDecoder, MLP, inverse_sigmoid,
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
IMAGE_PATH = "/home/yito/datasets/coco/val2017/000000000139.jpg"


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
    return tensor.unsqueeze(0), mask.unsqueeze(0), torch.tensor([[orig_h, orig_w]])


def load_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in sd.items()}


def build_prediction_heads(state_dict):
    last = N_DEC_LAYERS - 1
    class_embed = nn.Linear(D_MODEL, NUM_CLASSES)
    class_embed.load_state_dict({k.replace(f"class_embed.{last}.", ""): v
        for k, v in state_dict.items() if k.startswith(f"class_embed.{last}.")})
    class_embed.eval()
    bbox_embed = MLP(D_MODEL, D_MODEL, 4, 3)
    bbox_embed.load_state_dict({k.replace(f"bbox_embed.{last}.", ""): v
        for k, v in state_dict.items() if k.startswith(f"bbox_embed.{last}.")})
    bbox_embed.eval()
    return class_embed, bbox_embed


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

    use_device_backbone = os.environ.get("EDPOSE_DEVICE_BACKBONE", "1") == "1"
    print(f"Building pipeline (backbone={'device' if use_device_backbone else 'CPU'})...")
    t0 = time.time()
    if use_device_backbone:
        backbone = TTSwinLBackbone(device, CHECKPOINT_PATH)
    else:
        backbone = TTSwinBackbone(device, CHECKPOINT_PATH, use_compile=True)
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    decoder = TTEDPoseDecoder(
        device, full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
        N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES, NUM_BODY_POINTS,
        NUM_BOX_DEC_LAYERS, NUM_GROUP)
    class_embed, bbox_embed = build_prediction_heads(full_sd)
    attn_mask2 = prepare_attn_mask2()
    print(f"Built in {time.time() - t0:.1f}s\n")

    tensor, mask, orig_size = preprocess_image(IMAGE_PATH)
    print(f"Image: {os.path.basename(IMAGE_PATH)}, padded: {tensor.shape[2]}x{tensor.shape[3]}\n")

    n_runs = 5 if not use_device_backbone else int(os.environ.get("EDPOSE_RUNS", "1"))
    all_timings = []

    for i in range(n_runs):
        ti = {}

        t = time.time()
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        ti["backbone"] = time.time() - t

        t = time.time()
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        with torch.no_grad():
            enc_out_tt = encoder(src_tt, pos_tt, bb_out["reference_points"],
                bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"])
        memory = ttnn.to_torch(enc_out_tt).float()
        ttnn.deallocate(src_tt)
        ttnn.deallocate(pos_tt)
        ti["encoder"] = time.time() - t

        t = time.time()
        with torch.no_grad():
            query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
        ti["two_stage"] = time.time() - t

        t = time.time()
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
        ti["decoder"] = time.time() - t

        t = time.time()
        with torch.no_grad():
            last_hs = hs[-1]
            stride = NUM_BODY_POINTS + 1
            hs_bbox = last_hs[:, 0::stride, :]
            pred_logits = class_embed(hs_bbox)
            scores = pred_logits.sigmoid().max(-1)[0].squeeze(0)
            n_det = (scores > 0.3).sum().item()
        ti["heads"] = time.time() - t

        ti["total"] = sum(ti.values())
        all_timings.append(ti)

        label = "COLD" if i == 0 else "WARM"
        print(f"[{i+1}/{n_runs}] bb={ti['backbone']*1000:>5.0f}  "
              f"enc={ti['encoder']*1000:>6.0f}  "
              f"ts={ti['two_stage']*1000:>4.0f}  "
              f"dec={ti['decoder']*1000:>6.0f}  "
              f"hd={ti['heads']*1000:>3.0f}  "
              f"total={ti['total']*1000:>6.0f}ms  "
              f"det={n_det}  [{label}]")

    if len(all_timings) > 1:
        warm = all_timings[1:]
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"Average of runs 2-{n_runs} (warm, {len(warm)} runs), same image:")
        print(sep)
        for key in ["backbone", "encoder", "two_stage", "decoder", "heads", "total"]:
            vals = [t[key] for t in warm]
            avg = sum(vals) / len(vals)
            print(f"  {key:>12}: {avg*1000:>7.0f}ms")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
