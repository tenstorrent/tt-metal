# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify optimizations: proposals cache correctness, encoder determinism, two-stage consistency."""

import os
import sys
import time

import torch

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoder
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import (
    TwoStageQueryGenerator,
    gen_encoder_output_proposals,
)

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")
D_MODEL, D_FFN, N_HEADS, N_LEVELS, N_POINTS = 256, 2048, 8, 5, 4
N_ENC_LAYERS = 6
NUM_QUERIES, NUM_CLASSES = 900, 2


def pcc(a, b):
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    a_m = a_f - a_f.mean()
    b_m = b_f - b_f.mean()
    return ((a_m * b_m).sum() / (a_m.norm() * b_m.norm()).clamp(min=1e-8)).item()


@torch.no_grad()
def test_proposals_cache():
    """Verify cached proposals match gen_encoder_output_proposals for no-padding case."""
    print("=== Test 1: Proposals caching ===")
    spatial_shapes = torch.tensor([[200, 304], [100, 152], [50, 76], [25, 38], [13, 19]], dtype=torch.long)
    Lq = sum(int(H) * int(W) for H, W in spatial_shapes)
    memory = torch.randn(1, Lq, D_MODEL)
    mask_flatten = torch.zeros(1, Lq, dtype=torch.bool)

    # Reference: full gen_encoder_output_proposals
    ref_memory, ref_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

    # Cached version: simulate the fast-path
    ts = TwoStageQueryGenerator.__new__(TwoStageQueryGenerator)
    ts._cached_proposals = None
    ts._cached_valid_mask = None
    ts._cache_proposals(spatial_shapes)
    cached_proposals = ts._cached_proposals
    cached_memory = memory * ts._cached_valid_mask.float()

    # Compare proposals (finite values only, inf positions should match)
    finite_mask = torch.isfinite(ref_proposals) & torch.isfinite(cached_proposals)
    inf_match = (torch.isinf(ref_proposals) == torch.isinf(cached_proposals)).all().item()
    finite_match = torch.allclose(
        cached_proposals[finite_mask], ref_proposals[finite_mask], atol=1e-6
    )
    memory_match = torch.allclose(cached_memory, ref_memory, atol=1e-6)

    n_invalid = (~ts._cached_valid_mask).sum().item()
    print(f"  Proposals inf positions match: {inf_match}")
    print(f"  Proposals finite values match: {finite_match}")
    print(f"  Memory (masked) match: {memory_match}")
    print(f"  Invalid tokens (edge): {n_invalid}/{Lq}")

    all_pass = inf_match and finite_match and memory_match
    print(f"  {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


@torch.no_grad()
def test_two_stage_consistency():
    """Verify optimized two-stage produces same output as non-cached version."""
    print("=== Test 2: Two-Stage fast-path consistency ===")
    full_sd = {}
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    full_sd = {k.replace("module.", ""): v for k, v in sd.items()}

    ts_cached = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    ts_nocache = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    ts_nocache._cached_proposals = None  # force full computation

    spatial_shapes = torch.tensor([[200, 304], [100, 152], [50, 76], [25, 38], [13, 19]], dtype=torch.long)
    Lq = sum(int(H) * int(W) for H, W in spatial_shapes)
    memory = torch.randn(1, Lq, D_MODEL)
    mask = torch.zeros(1, Lq, dtype=torch.bool)

    # Warm up cache
    ts_cached(memory, mask, spatial_shapes)

    out_cached = ts_cached(memory, mask, spatial_shapes)
    out_nocache = ts_nocache(memory, mask, spatial_shapes)

    results = {}
    for key in ["tgt", "refpoint_embed", "init_box_proposal"]:
        p = pcc(out_cached[key], out_nocache[key])
        results[key] = p
        print(f"  {key}: PCC = {p:.8f}")

    all_pass = all(v > 0.999 for v in results.values())
    print(f"  {'PASS' if all_pass else 'FAIL'}")
    print()
    return all_pass


@torch.no_grad()
def test_encoder_determinism():
    """Verify encoder produces consistent output across two runs."""
    print("=== Test 3: Encoder determinism ===")
    device = ttnn.open_device(device_id=0)
    full_sd = {}
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    full_sd = {k.replace("module.", ""): v for k, v in sd.items()}

    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN,
                                   N_LEVELS, N_HEADS, N_POINTS)

    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    torch.manual_seed(42)
    image = torch.randn(1, 3, 800, 1216)
    mask = torch.zeros(1, 800, 1216, dtype=torch.bool)
    bb_out = backbone(image, mask)

    src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Run 1
    t0 = time.time()
    out1_tt = encoder(src_tt, pos_tt, bb_out["reference_points"],
                       bb_out["spatial_shapes"], bb_out["level_start_index"],
                       bb_out["mask_flatten"])
    out1 = ttnn.to_torch(out1_tt).float()
    ttnn.deallocate(out1_tt)
    print(f"  Run 1: {(time.time()-t0)*1000:.0f}ms, shape={out1.shape}")

    # Run 2
    t0 = time.time()
    out2_tt = encoder(src_tt, pos_tt, bb_out["reference_points"],
                       bb_out["spatial_shapes"], bb_out["level_start_index"],
                       bb_out["mask_flatten"])
    out2 = ttnn.to_torch(out2_tt).float()
    ttnn.deallocate(out2_tt)
    print(f"  Run 2: {(time.time()-t0)*1000:.0f}ms, shape={out2.shape}")

    det_pcc = pcc(out1, out2)
    print(f"  Determinism PCC: {det_pcc:.8f}")

    # Basic sanity: output should not be all zeros or NaN
    sane = not out1.isnan().any() and out1.abs().mean() > 0.01
    print(f"  Output sanity (no NaN, non-zero): {sane}")

    all_pass = det_pcc > 0.999 and sane
    print(f"  {'PASS' if all_pass else 'FAIL'}")

    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    ttnn.close_device(device)
    print()
    return all_pass


def main():
    pass1 = test_proposals_cache()
    pass2 = test_two_stage_consistency()
    pass3 = test_encoder_determinism()
    print(f"=== Overall: {'ALL PASS' if all([pass1, pass2, pass3]) else 'SOME FAILED'} ===")


if __name__ == "__main__":
    main()
