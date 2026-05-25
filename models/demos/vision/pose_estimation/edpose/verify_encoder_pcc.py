# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Verify encoder layer PCC after optimizations.

Captures reference output from current implementation, then compares
against new implementation. Both MSDeformAttn and full layer outputs.

Usage:
    python models/demos/vision/pose_estimation/edpose/verify_encoder_pcc.py [--save-ref]
"""

import argparse
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
REF_DIR = os.path.join(os.path.dirname(__file__), "pcc_refs")

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


def compute_pcc(ref, test):
    ref_f = ref.float().flatten()
    test_f = test.float().flatten()
    ref_f = ref_f - ref_f.mean()
    test_f = test_f - test_f.mean()
    num = (ref_f * test_f).sum()
    den = (ref_f.norm() * test_f.norm()).clamp(min=1e-8)
    return (num / den).item()


def run_layer0_attn(layer, src_tt, pos_tt, bb_out, normalizer,
                    cpu_ref_levels, cpu_norm_levels, device):
    query = ttnn.add(src_tt, pos_tt)
    attn_out = layer.self_attn(
        query, bb_out["reference_points"], src_tt, bb_out["spatial_shapes"],
        bb_out["level_start_index"], bb_out["mask_flatten"],
        normalizer=normalizer,
        cpu_ref_levels=cpu_ref_levels, cpu_norm_levels=cpu_norm_levels,
    )
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(attn_out).float()
    ttnn.deallocate(query)
    ttnn.deallocate(attn_out)
    return result


def run_layer0_full(layer, src_tt, pos_tt, bb_out, normalizer,
                    cpu_ref_levels, cpu_norm_levels, device):
    out = layer(
        src_tt, pos_tt, bb_out["reference_points"], bb_out["spatial_shapes"],
        bb_out["level_start_index"], bb_out["mask_flatten"],
        normalizer=normalizer,
        cpu_ref_levels=cpu_ref_levels, cpu_norm_levels=cpu_norm_levels,
    )
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out).float()
    ttnn.deallocate(out)
    return result


def run_full_encoder(encoder, src_tt, pos_tt, bb_out, device):
    enc_out = encoder(src_tt, pos_tt, bb_out["reference_points"],
        bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"])
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(enc_out).float()
    ttnn.deallocate(enc_out)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-ref", action="store_true", help="Save reference outputs")
    args = parser.parse_args()

    device = ttnn.open_device(device_id=0)
    full_sd = load_state_dict()

    print("Building backbone + encoder...")
    backbone = TTSwinLBackbone(device, CHECKPOINT_PATH)
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
    print("Built.\n")

    tensor, mask = preprocess_image(IMAGE_PATH)

    # Warm-up
    print("Warm-up...")
    with torch.no_grad():
        bb_out = backbone(tensor, mask)
    src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    enc_out = encoder(src_tt, pos_tt, bb_out["reference_points"],
        bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"])
    ttnn.deallocate(enc_out)
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    print("Warm-up done.\n")

    normalizer = torch.stack(
        [bb_out["spatial_shapes"][..., 1], bb_out["spatial_shapes"][..., 0]], -1
    ).float()
    L = len(bb_out["spatial_shapes"])
    ref_f = bb_out["reference_points"].float()
    cpu_ref_levels = [ref_f[:, :, lid, :].unsqueeze(2).unsqueeze(3) for lid in range(L)]
    cpu_norm_levels = [normalizer[lid].view(1, 1, 1, 1, 2) for lid in range(L)]

    # Re-run backbone to get fresh inputs
    with torch.no_grad():
        bb_out = backbone(tensor, mask)
    src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if args.save_ref:
        os.makedirs(REF_DIR, exist_ok=True)

        print("Saving reference: layer 0 attention output...")
        attn_ref = run_layer0_attn(encoder.layers[0], src_tt, pos_tt, bb_out,
                                   normalizer, cpu_ref_levels, cpu_norm_levels, device)
        torch.save(attn_ref, os.path.join(REF_DIR, "layer0_attn.pt"))
        print(f"  Shape: {attn_ref.shape}, mean: {attn_ref.mean():.6f}, std: {attn_ref.std():.6f}")

        # Need fresh src_tt since layer0 forward consumed the input
        ttnn.deallocate(pos_tt)
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print("Saving reference: layer 0 full output...")
        layer_ref = run_layer0_full(encoder.layers[0], src_tt, pos_tt, bb_out,
                                    normalizer, cpu_ref_levels, cpu_norm_levels, device)
        torch.save(layer_ref, os.path.join(REF_DIR, "layer0_full.pt"))
        print(f"  Shape: {layer_ref.shape}, mean: {layer_ref.mean():.6f}, std: {layer_ref.std():.6f}")

        ttnn.deallocate(pos_tt)
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        print("Saving reference: full encoder output...")
        enc_ref = run_full_encoder(encoder, src_tt, pos_tt, bb_out, device)
        torch.save(enc_ref, os.path.join(REF_DIR, "encoder_full.pt"))
        print(f"  Shape: {enc_ref.shape}, mean: {enc_ref.mean():.6f}, std: {enc_ref.std():.6f}")

        print(f"\nReferences saved to {REF_DIR}/")
        ttnn.deallocate(pos_tt)
    else:
        # Compare against saved references
        if not os.path.exists(REF_DIR):
            print(f"ERROR: No reference directory at {REF_DIR}. Run with --save-ref first.")
            ttnn.close_device(device)
            return

        all_pass = True

        print("=== PCC Verification ===\n")

        # Layer 0 attention
        attn_ref = torch.load(os.path.join(REF_DIR, "layer0_attn.pt"))
        t0 = time.time()
        attn_test = run_layer0_attn(encoder.layers[0], src_tt, pos_tt, bb_out,
                                    normalizer, cpu_ref_levels, cpu_norm_levels, device)
        t1 = time.time()
        pcc = compute_pcc(attn_ref, attn_test)
        status = "PASS" if pcc >= 0.99 else "FAIL"
        if pcc < 0.99:
            all_pass = False
        print(f"Layer 0 Attention: PCC={pcc:.6f} [{status}]  ({(t1-t0)*1000:.0f}ms)")

        # Layer 0 full
        ttnn.deallocate(pos_tt)
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        layer_ref = torch.load(os.path.join(REF_DIR, "layer0_full.pt"))
        t0 = time.time()
        layer_test = run_layer0_full(encoder.layers[0], src_tt, pos_tt, bb_out,
                                     normalizer, cpu_ref_levels, cpu_norm_levels, device)
        t1 = time.time()
        pcc = compute_pcc(layer_ref, layer_test)
        status = "PASS" if pcc >= 0.99 else "FAIL"
        if pcc < 0.99:
            all_pass = False
        print(f"Layer 0 Full:      PCC={pcc:.6f} [{status}]  ({(t1-t0)*1000:.0f}ms)")

        # Full encoder
        ttnn.deallocate(pos_tt)
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        enc_ref = torch.load(os.path.join(REF_DIR, "encoder_full.pt"))
        t0 = time.time()
        enc_test = run_full_encoder(encoder, src_tt, pos_tt, bb_out, device)
        t1 = time.time()
        pcc = compute_pcc(enc_ref, enc_test)
        status = "PASS" if pcc >= 0.99 else "FAIL"
        if pcc < 0.99:
            all_pass = False
        print(f"Full Encoder:      PCC={pcc:.6f} [{status}]  ({(t1-t0)*1000:.0f}ms)")

        print(f"\n{'ALL PASS' if all_pass else 'SOME FAILED'}")
        ttnn.deallocate(pos_tt)

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
