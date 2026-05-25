# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test: backbone + encoder x3 to isolate hang."""

import os
import sys
import time

import torch
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

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")
IMAGE_PATH = "/home/yito/datasets/coco/val2017/000000000139.jpg"

D_MODEL = 256
D_FFN = 2048
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4
N_ENC_LAYERS = 6


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


def main():
    device = ttnn.open_device(device_id=0)
    full_sd = load_state_dict()

    print("Building backbone + encoder...")
    backbone = TTSwinLBackbone(device, CHECKPOINT_PATH)
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)

    tensor, mask = preprocess_image(IMAGE_PATH)
    print(f"Image padded: {tensor.shape[2]}x{tensor.shape[3]}\n")

    for i in range(3):
        print(f"--- Run {i+1}/3 ---")

        t = time.time()
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        t_bb = time.time() - t
        print(f"  backbone: {t_bb*1000:.0f}ms")

        t = time.time()
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"  transferred src/pos to device")

        with torch.no_grad():
            enc_out_tt = encoder(src_tt, pos_tt, bb_out["reference_points"],
                bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"])
        t_enc = time.time() - t
        print(f"  encoder: {t_enc*1000:.0f}ms")

        memory = ttnn.to_torch(enc_out_tt).float()
        print(f"  memory shape: {memory.shape}")

        ttnn.deallocate(src_tt)
        ttnn.deallocate(pos_tt)
        ttnn.deallocate(enc_out_tt)
        print(f"  deallocated src/pos/enc_out\n")

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
