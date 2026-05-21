# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test TTSwinBackbone against CPU reference."""

import os
import sys
import time

import torch

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_swin_backbone import TTSwinBackbone

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")


def pcc(a, b):
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    num = (a_centered * b_centered).sum()
    denom = (a_centered.norm() * b_centered.norm()) + 1e-8
    return (num / denom).item()


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.\n")

    print("Loading CPU reference backbone...")
    t0 = time.time()
    ref_backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    print(f"CPU backbone loaded in {time.time()-t0:.1f}s\n")

    print("Loading TTSwinBackbone...")
    t0 = time.time()
    tt_backbone = TTSwinBackbone(device, CHECKPOINT_PATH)
    print(f"TT backbone loaded in {time.time()-t0:.1f}s\n")

    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    # CPU reference
    print("Running CPU reference...")
    t0 = time.time()
    with torch.no_grad():
        ref_out = ref_backbone(image_tensor, mask)
    t_ref = time.time() - t0
    print(f"CPU backbone: {t_ref*1000:.0f}ms\n")

    # TT backbone
    print("Running TT backbone (cold)...")
    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_backbone(image_tensor, mask)
    t_cold = time.time() - t0
    print(f"TT backbone (cold): {t_cold*1000:.0f}ms\n")

    print("Running TT backbone (warm)...")
    t0 = time.time()
    with torch.no_grad():
        tt_out = tt_backbone(image_tensor, mask)
    t_warm = time.time() - t0
    print(f"TT backbone (warm): {t_warm*1000:.0f}ms\n")

    # Compare outputs
    print("=" * 60)
    print("Output comparison (TT vs CPU reference):")
    print("=" * 60)

    for key in ["src_flatten", "pos_flatten", "spatial_shapes", "level_start_index",
                "mask_flatten", "reference_points"]:
        ref_val = ref_out[key]
        tt_val = tt_out[key]

        if key in ("spatial_shapes", "level_start_index"):
            match = torch.equal(ref_val, tt_val)
            print(f"  {key:>20s}: {'MATCH' if match else 'MISMATCH'}")
            if not match:
                print(f"    ref: {ref_val}")
                print(f"    tt:  {tt_val}")
        elif key == "mask_flatten":
            match = torch.equal(ref_val, tt_val)
            print(f"  {key:>20s}: {'MATCH' if match else 'MISMATCH'}")
        else:
            p = pcc(ref_val, tt_val)
            status = "PASS" if p > 0.98 else "FAIL"
            print(f"  {key:>20s}: PCC={p:.5f} {status}")

    # Per-level src PCC
    print("\nPer-level src PCC:")
    spatial_shapes = ref_out["spatial_shapes"]
    level_start = ref_out["level_start_index"]
    for lvl in range(len(spatial_shapes)):
        start = int(level_start[lvl])
        h, w = int(spatial_shapes[lvl][0]), int(spatial_shapes[lvl][1])
        end = start + h * w
        ref_slice = ref_out["src_flatten"][:, start:end, :]
        tt_slice = tt_out["src_flatten"][:, start:end, :]
        p = pcc(ref_slice, tt_slice)
        print(f"  Level {lvl} ({h}x{w}): PCC={p:.5f}")

    print(f"\nTiming: CPU={t_ref*1000:.0f}ms  TT(cold)={t_cold*1000:.0f}ms  TT(warm)={t_warm*1000:.0f}ms")
    print(f"Speedup: {t_ref/t_warm:.1f}x")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
