# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
import time
from pathlib import Path

import torch

import ttnn

repo_path = str(Path.cwd() / "RT-DETR" / "rtdetr_pytorch")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from src.core import YAMLConfig
from tt.rtdetr_encoder import run_encoder
from tt.weight_utils import get_tt_parameters


def run_benchmark():
    config_path = "RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r50vd_6x_coco.yml"
    ckpt_path = "weights/rtdetr_r50vd.pth"

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        print("Loading Model & Weights...")
        cfg = YAMLConfig(config_path)
        model = cfg.model
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["ema"]["module"])
        model.eval()

        print("Converting weights to BFLOAT8_B and pushing to Device...")
        tt_params = get_tt_parameters(device, model)

        src = torch.randn(1, 1, 300, 256)
        pos = torch.randn(1, 1, 300, 256)

        print("Warming up...")
        for _ in range(3):
            _ = run_encoder(src, tt_params.layers, device, pos_embed=pos)
        ttnn.synchronize_device(device)

        print("Benchmarking (50 iterations)...")
        start_time = time.time()
        for _ in range(50):
            _ = run_encoder(src, tt_params.layers, device, pos_embed=pos)
        ttnn.synchronize_device(device)
        end_time = time.time()

        total_time = end_time - start_time
        fps = 50 / total_time
        latency = (total_time / 50) * 1000

        print("### Performance Report")
        print("| Model | Batch Size | Hardware | Precision/Config | Actual FPS | Latency (ms) |")
        print("|---|---|---|---|---|---|")
        print(
            f"| RT-DETR (Encoder) | 1 | Wormhole (N300) | BFP8 Weights, L1 Interleaved, Fused GELU | **{fps:.2f}** | **{latency:.2f}** |"
        )

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    run_benchmark()
