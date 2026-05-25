# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile encoder layer: per-op timing breakdown for one warm layer."""

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
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_ms_deform_attn import TTMSDeformAttn

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


def profile_msdeform_attn(attn_module, query_tt, src_tt, reference_points,
                          spatial_shapes, level_start_index, mask_flatten,
                          normalizer, cpu_ref_levels, cpu_norm_levels, device):
    """Profile MSDeformAttn internals for one call."""
    M = attn_module.n_heads
    D = attn_module.d_per_head
    L = attn_module.n_levels
    P = attn_module.n_points

    t0 = time.time()
    value_tt = ttnn.linear(src_tt, attn_module.value_proj_w, bias=attn_module.value_proj_b)
    offsets_tt = ttnn.linear(query_tt, attn_module.sampling_offsets_w, bias=attn_module.sampling_offsets_b)
    attn_w_tt = ttnn.linear(query_tt, attn_module.attention_weights_w, bias=attn_module.attention_weights_b)
    ttnn.synchronize_device(device)
    t_linears = time.time() - t0

    t0 = time.time()
    offsets_t = ttnn.to_torch(offsets_tt).float()
    t_offsets_transfer = time.time() - t0

    ttnn.deallocate(offsets_tt)
    N = 1
    Lq = offsets_t.shape[1]
    offsets_t = offsets_t.view(N, Lq, M, L, P, 2)

    t0 = time.time()
    attn_w_tt_proc, _, _ = attn_module._compute_attn_weights(attn_w_tt, M, L, P)
    ttnn.synchronize_device(device)
    t_attn_weights = time.time() - t0

    if mask_flatten is not None:
        inv_mask = (~mask_flatten).float().unsqueeze(-1).to(torch.bfloat16)
        inv_mask_tt = ttnn.from_torch(inv_mask, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        old = value_tt
        value_tt = ttnn.multiply(value_tt, inv_mask_tt)
        ttnn.deallocate(old)
        ttnn.deallocate(inv_mask_tt)

    old = value_tt
    value_tt = ttnn.to_layout(value_tt, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(old)
    Len_in = list(value_tt.shape)[1]
    value_tt = ttnn.reshape(value_tt, (N, Len_in, M, D))

    t_grid_compute = 0
    t_grid_transfer = 0
    t_grid_sample = 0
    t_weighted_sum = 0

    result_accum = None
    for lid, (H, W) in enumerate(spatial_shapes):
        H, W = int(H), int(W)
        start = int(level_start_index[lid])
        end = start + H * W

        val = ttnn.slice(value_tt, [0, start, 0, 0], [N, end, M, D])
        val = ttnn.transpose(val, 1, 2)
        val = ttnn.reshape(val, (N * M, H, W, D))

        tg0 = time.time()
        off_l = offsets_t[:, :, :, lid, :, :]
        precomputed_tt = attn_module._fused_grid_for_level(
            off_l, cpu_ref_levels[lid], cpu_norm_levels[lid], H, W, N * M, Lq, P)
        tg1 = time.time()
        t_grid_compute += (tg1 - tg0)

        ts0 = time.time()
        sampled = ttnn.grid_sample(val, precomputed_tt, use_precomputed_grid=True, align_corners=False)
        ttnn.synchronize_device(device)
        ts1 = time.time()
        ttnn.deallocate(val)
        ttnn.deallocate(precomputed_tt)
        t_grid_sample += (ts1 - ts0)

        tw0 = time.time()
        attn_l_tt = ttnn.slice(attn_w_tt_proc, [0, 0, lid * P, 0], [N * M, Lq, (lid + 1) * P, 1])
        weighted = ttnn.multiply(attn_l_tt, sampled)
        ttnn.deallocate(attn_l_tt)
        ttnn.deallocate(sampled)
        if result_accum is None:
            result_accum = weighted
        else:
            old = result_accum
            result_accum = ttnn.add(result_accum, weighted)
            ttnn.deallocate(old)
            ttnn.deallocate(weighted)
        ttnn.synchronize_device(device)
        tw1 = time.time()
        t_weighted_sum += (tw1 - tw0)

    t0 = time.time()
    old = result_accum
    result_accum = ttnn.sum(result_accum, dim=2)
    ttnn.deallocate(old)
    old = result_accum
    result = ttnn.to_layout(result_accum, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(old)
    result = ttnn.reshape(result, (N, M, Lq, D))
    result = ttnn.transpose(result, 1, 2)
    result = ttnn.reshape(result, (N, Lq, M * D))
    old = result
    result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
    ttnn.deallocate(old)
    output = ttnn.linear(result, attn_module.output_proj_w, bias=attn_module.output_proj_b)
    ttnn.deallocate(result)
    ttnn.synchronize_device(device)
    t_output = time.time() - t0

    ttnn.deallocate(value_tt)
    ttnn.deallocate(attn_w_tt_proc)
    ttnn.deallocate(output)

    return {
        "linears": t_linears,
        "offsets_transfer": t_offsets_transfer,
        "attn_weights": t_attn_weights,
        "grid_compute": t_grid_compute,
        "grid_sample": t_grid_sample,
        "weighted_sum": t_weighted_sum,
        "output_proj": t_output,
    }


def main():
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

    # Run full pipeline once to warm JIT
    print("Warm-up run (backbone + encoder)...")
    with torch.no_grad():
        bb_out = backbone(tensor, mask)
    src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    with torch.no_grad():
        enc_out_tt = encoder(src_tt, pos_tt, bb_out["reference_points"],
            bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"])
    ttnn.deallocate(enc_out_tt)
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    print("Warm-up done.\n")

    # Now profile a single encoder layer in detail
    print("Profiling encoder layer 0 (warm)...")
    with torch.no_grad():
        bb_out = backbone(tensor, mask)
    src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    normalizer = torch.stack(
        [bb_out["spatial_shapes"][..., 1], bb_out["spatial_shapes"][..., 0]], -1
    ).float()
    L = len(bb_out["spatial_shapes"])
    ref_f = bb_out["reference_points"].float()
    cpu_ref_levels = [ref_f[:, :, lid, :].unsqueeze(2).unsqueeze(3) for lid in range(L)]
    cpu_norm_levels = [normalizer[lid].view(1, 1, 1, 1, 2) for lid in range(L)]

    layer0 = encoder.layers[0]
    query = ttnn.add(src_tt, pos_tt)

    timings = profile_msdeform_attn(
        layer0.self_attn, query, src_tt, bb_out["reference_points"],
        bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"],
        normalizer, cpu_ref_levels, cpu_norm_levels, device)

    ttnn.deallocate(query)

    print("\n=== MSDeformAttn Layer 0 Breakdown ===")
    total = sum(timings.values())
    for k, v in timings.items():
        print(f"  {k:>20}: {v*1000:>7.1f}ms  ({v/total*100:>5.1f}%)")
    print(f"  {'TOTAL':>20}: {total*1000:>7.1f}ms")

    print(f"\nFYI: Encoder has {N_ENC_LAYERS} layers, so extrapolated total ≈ {total*N_ENC_LAYERS*1000:.0f}ms")
    print(f"Measured warm encoder total was ~2707ms → per-layer ~{2707/N_ENC_LAYERS:.0f}ms")

    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)
    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
