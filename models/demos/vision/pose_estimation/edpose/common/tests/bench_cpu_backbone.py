# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark CPU backbone with various optimizations."""

import os
import sys
import time

import torch

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")


def bench(label, backbone, image_tensor, mask, n_warmup=1, n_iter=3):
    for _ in range(n_warmup):
        with torch.inference_mode():
            _ = backbone(image_tensor, mask)

    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = backbone(image_tensor, mask)
        times.append(time.perf_counter() - t0)
    avg = sum(times) / len(times)
    best = min(times)
    print(f"  {label}: avg={avg*1000:.0f}ms  best={best*1000:.0f}ms")
    return best


def main():
    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    print("Loading backbone...")
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")

    print(f"\nDefault threads: {torch.get_num_threads()}")

    # Baseline
    bench("Baseline (no_grad)", backbone, image_tensor, mask)

    # Try different thread counts
    for n_threads in [8, 16, 24, 32]:
        torch.set_num_threads(n_threads)
        bench(f"Threads={n_threads}", backbone, image_tensor, mask)

    # Reset to best
    torch.set_num_threads(16)

    # Try torch.compile on Swin
    print("\nAttempting torch.compile on Swin...")
    try:
        backbone.swin = torch.compile(backbone.swin, mode="reduce-overhead")
        bench("torch.compile(swin, reduce-overhead)", backbone, image_tensor, mask,
              n_warmup=3, n_iter=3)
    except Exception as e:
        print(f"  torch.compile failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
