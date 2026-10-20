# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precompute the MSE-prepared bfloat4_b expert weights on the host, without a device.

Converting the weight cache dequantizes 256 MXFP4 experts per layer and prepares them (tt.common.bfp_mse_prepare,
~80 s per layer at 32 threads). This writes each layer's (w_gate_up, w_down) in device layout to
<out>/layer<i>.pt so the cache conversion (MIMO_EXPERT_PREP_DIR=<out>) only encodes them.

    python models/experimental/mimo_v2_6_flash/tests/perf/prep_experts.py --out DIR [--layers 1-47] [--threads 32]
"""

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tt.moe import expert_weights_host


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="1-47", help="a-b or comma list")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--tp", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if "-" in args.layers:
        a, b = map(int, args.layers.split("-"))
        layers = range(a, b + 1)
    else:
        layers = [int(x) for x in args.layers.split(",")]
    os.makedirs(args.out, exist_ok=True)
    cfg = MiMoConfig.from_model_path(default_model_path())
    ckpt = MiMoCheckpoint(default_model_path(), cfg)
    inter_local = cfg.moe_intermediate_size // args.tp
    for i in layers:
        path = os.path.join(args.out, f"layer{i}.pt")
        if os.path.exists(path):
            continue
        t0 = time.time()
        experts = ckpt.experts(i)
        w_gate_up, w_down = expert_weights_host(experts, args.tp, inter_local, ttnn.bfloat4_b)
        del experts
        torch.save({"w_gate_up": w_gate_up, "w_down": w_down}, path + ".tmp")
        os.replace(path + ".tmp", path)
        logger.info(f"layer {i}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
