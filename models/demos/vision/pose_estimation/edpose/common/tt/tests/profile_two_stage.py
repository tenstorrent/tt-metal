# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile Two-Stage sub-op timing."""

import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import (
    TwoStageQueryGenerator,
    gen_encoder_output_proposals,
)

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")
D_MODEL = 256
NUM_QUERIES = 900
NUM_CLASSES = 2


def load_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in sd.items()}


@torch.no_grad()
def main():
    full_sd = load_state_dict()
    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)

    spatial_shapes = torch.tensor([[100, 152], [50, 76], [25, 38], [13, 19], [7, 10]], dtype=torch.long)
    Lq = sum(int(H) * int(W) for H, W in spatial_shapes)
    memory = torch.randn(1, Lq, D_MODEL)
    mask_flatten = torch.zeros(1, Lq, dtype=torch.bool)

    # Warm-up
    for _ in range(2):
        two_stage(memory, mask_flatten, spatial_shapes)

    N_RUNS = 5
    timings = {}

    for _ in range(N_RUNS):
        t = time.perf_counter()
        output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        dt = time.perf_counter() - t
        timings.setdefault("gen_proposals", []).append(dt)

        t = time.perf_counter()
        output_memory = two_stage.enc_output(output_memory)
        dt = time.perf_counter() - t
        timings.setdefault("enc_output_linear", []).append(dt)

        t = time.perf_counter()
        output_memory = two_stage.enc_output_norm(output_memory)
        dt = time.perf_counter() - t
        timings.setdefault("enc_output_norm", []).append(dt)

        t = time.perf_counter()
        enc_outputs_class = two_stage.enc_out_class_embed(output_memory)
        dt = time.perf_counter() - t
        timings.setdefault("class_embed", []).append(dt)

        t = time.perf_counter()
        enc_outputs_coord = two_stage.enc_out_bbox_embed(output_memory) + output_proposals
        dt = time.perf_counter() - t
        timings.setdefault("bbox_embed", []).append(dt)

        t = time.perf_counter()
        topk_proposals = torch.topk(enc_outputs_class.max(-1)[0], NUM_QUERIES, dim=1)[1]
        dt = time.perf_counter() - t
        timings.setdefault("topk", []).append(dt)

        t = time.perf_counter()
        refpoint_embed = torch.gather(enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).detach()
        init_box_proposal = torch.gather(output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid()
        tgt = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, D_MODEL)).detach()
        hs_enc = torch.gather(output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, D_MODEL)).unsqueeze(0)
        ref_enc = torch.gather(enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).sigmoid().unsqueeze(0)
        dt = time.perf_counter() - t
        timings.setdefault("gather_outputs", []).append(dt)

    print("=== Two-Stage sub-op profiling ===\n")
    total_avg = 0
    for name in ["gen_proposals", "enc_output_linear", "enc_output_norm", "class_embed",
                  "bbox_embed", "topk", "gather_outputs"]:
        vals = timings[name]
        avg = sum(vals) / len(vals) * 1000
        total_avg += avg
        print(f"  {name:>25s}: {avg:>7.1f}ms")
    print(f"\n  {'TOTAL':>25s}: {total_avg:>7.1f}ms")
    print()


if __name__ == "__main__":
    main()
