# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify device Swin-L backbone matches CPU reference (PCC ≥ 0.99)."""

import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

TT_METAL_HOME = os.environ.get(
    "TT_METAL_HOME",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")),
)
if TT_METAL_HOME not in sys.path:
    sys.path.insert(0, TT_METAL_HOME)
os.environ.setdefault("EDPOSE_ROOT", os.path.expanduser("~/ttwork/ED-Pose"))

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_swin_backbone import (
    TTSwinBackbone,
    TTSwinLBackbone,
)

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
    return tensor.unsqueeze(0), mask.unsqueeze(0)


def pcc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_mean = a_flat - a_flat.mean()
    b_mean = b_flat - b_flat.mean()
    num = (a_mean * b_mean).sum()
    den = (a_mean.pow(2).sum() * b_mean.pow(2).sum()).sqrt()
    return (num / den).item() if den > 0 else 0.0


def main():
    device = ttnn.open_device(device_id=0)
    CHECKPOINT_PATH = os.path.join(os.environ["EDPOSE_ROOT"], "weights", "edpose_swinl_5scale_coco.pth")

    print("Building CPU backbone...")
    t0 = time.time()
    cpu_backbone = TTSwinBackbone(device, CHECKPOINT_PATH, use_compile=False)
    print(f"  CPU backbone built in {time.time() - t0:.1f}s")

    print("Building device backbone...")
    t0 = time.time()
    dev_backbone = TTSwinLBackbone(device, CHECKPOINT_PATH)
    print(f"  Device backbone built in {time.time() - t0:.1f}s")

    tensor, mask = preprocess_image(IMAGE_PATH)
    print(f"\nImage: {os.path.basename(IMAGE_PATH)}, shape: {tensor.shape}")

    print("\nRunning CPU backbone...")
    t0 = time.time()
    with torch.no_grad():
        cpu_out = cpu_backbone(tensor, mask)
    cpu_time = time.time() - t0
    print(f"  CPU backbone: {cpu_time*1000:.0f}ms")

    print("Running device backbone...")
    t0 = time.time()
    dev_out = dev_backbone(tensor, mask)
    dev_time = time.time() - t0
    print(f"  Device backbone: {dev_time*1000:.0f}ms")

    print(f"\n{'='*50}")
    print("Numerical comparison (PCC):")
    print(f"{'='*50}")

    for key in ["src_flatten", "pos_flatten"]:
        cpu_val = cpu_out[key]
        dev_val = dev_out[key]
        p = pcc(cpu_val, dev_val)
        status = "PASS" if p >= 0.99 else "FAIL"
        print(f"  {key:>20}: PCC={p:.6f}  [{status}]")
        print(f"    shapes: cpu={list(cpu_val.shape)} dev={list(dev_val.shape)}")

    for key in ["spatial_shapes", "level_start_index"]:
        match = torch.equal(cpu_out[key], dev_out[key])
        status = "PASS" if match else "FAIL"
        print(f"  {key:>20}: exact_match={match}  [{status}]")

    for key in ["mask_flatten"]:
        match = torch.equal(cpu_out[key], dev_out[key])
        status = "PASS" if match else "FAIL"
        print(f"  {key:>20}: exact_match={match}  [{status}]")

    ref_pcc = pcc(cpu_out["reference_points"], dev_out["reference_points"])
    status = "PASS" if ref_pcc >= 0.999 else "FAIL"
    print(f"  {'reference_points':>20}: PCC={ref_pcc:.6f}  [{status}]")

    vr_pcc = pcc(cpu_out["valid_ratios"], dev_out["valid_ratios"])
    status = "PASS" if vr_pcc >= 0.999 else "FAIL"
    print(f"  {'valid_ratios':>20}: PCC={vr_pcc:.6f}  [{status}]")

    print(f"\nSpeedup: {cpu_time/dev_time:.1f}x ({cpu_time*1000:.0f}ms → {dev_time*1000:.0f}ms)")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
