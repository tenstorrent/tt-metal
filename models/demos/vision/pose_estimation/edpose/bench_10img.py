# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark ED-Pose on 10 different COCO images, report warm-run average."""

import glob
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

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_swin_backbone import TTSwinBackbone
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


def preprocess_image(image_path, max_size=1333, target_size=800):
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = target_size / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > max_size:
        scale = max_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
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
def apply_prediction_heads(hs, references, class_embed, bbox_embed, pose_embed, kpt_index):
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
    return n_det


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


def main():
    coco_dir = "/home/yito/datasets/coco"
    images = sorted(glob.glob(os.path.join(coco_dir, "val2017", "*.jpg")))[:10]
    assert len(images) == 10, f"Need 10 images, found {len(images)}"

    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    full_sd = load_state_dict()

    print("Building pipeline...")
    t0 = time.time()
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
    class_embed, bbox_embed, pose_embed, kpt_index = build_prediction_heads(full_sd)
    attn_mask2 = prepare_attn_mask2()
    print(f"Pipeline built in {time.time() - t0:.1f}s\n")

    all_timings = []

    for i, img_path in enumerate(images):
        fname = os.path.basename(img_path)
        timings = {}

        tensor, mask, orig_size = preprocess_image(img_path)
        padded = f"{tensor.shape[2]}x{tensor.shape[3]}"

        t = time.time()
        bb_out = backbone(tensor, mask)
        timings["backbone"] = time.time() - t

        t = time.time()
        src_tt = ttnn.from_torch(
            bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(
            bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        enc_out_tt = encoder(
            src_tt, pos_tt, bb_out["reference_points"], bb_out["spatial_shapes"],
            bb_out["level_start_index"], bb_out["mask_flatten"])
        memory = ttnn.to_torch(enc_out_tt).float()
        ttnn.deallocate(src_tt)
        ttnn.deallocate(pos_tt)
        timings["encoder"] = time.time() - t

        t = time.time()
        query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
        timings["two_stage"] = time.time() - t

        t = time.time()
        hs, references = decoder(
            tgt=query_out["tgt"], memory_tt=enc_out_tt,
            refpoint_embed=query_out["refpoint_embed"],
            spatial_shapes=bb_out["spatial_shapes"],
            level_start_index=bb_out["level_start_index"],
            valid_ratios=bb_out["valid_ratios"],
            memory_key_padding_mask=bb_out["mask_flatten"],
            self_attn_mask=None, self_attn_mask2=attn_mask2)
        ttnn.deallocate(enc_out_tt)
        timings["decoder"] = time.time() - t

        t = time.time()
        n_det = apply_prediction_heads(hs, references, class_embed, bbox_embed, pose_embed, kpt_index)
        timings["heads"] = time.time() - t

        timings["total"] = sum(timings.values())
        all_timings.append(timings)

        label = "COLD" if i == 0 else f"WARM"
        print(f"[{i+1:>2}/10] {fname}  pad={padded}  det={n_det:>2}  "
              f"bb={timings['backbone']*1000:>5.0f}  enc={timings['encoder']*1000:>6.0f}  "
              f"ts={timings['two_stage']*1000:>4.0f}  dec={timings['decoder']*1000:>6.0f}  "
              f"hd={timings['heads']*1000:>3.0f}  total={timings['total']*1000:>6.0f}ms  [{label}]")

    # Average of runs 2-10 (warm)
    warm = all_timings[1:]
    print(f"\n{'='*70}")
    print(f"Average of runs 2-10 (warm, {len(warm)} runs):")
    print(f"{'='*70}")
    for key in ["backbone", "encoder", "two_stage", "decoder", "heads", "total"]:
        vals = [t[key] for t in warm]
        avg = sum(vals) / len(vals)
        mn = min(vals)
        mx = max(vals)
        print(f"  {key:>12}: avg={avg*1000:>7.0f}ms  min={mn*1000:>7.0f}ms  max={mx*1000:>7.0f}ms")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
