# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile ED-Pose encoder: per-op timing for one MSDeformAttn layer."""

import os
import sys
import time
from collections import defaultdict

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

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_swin_backbone import TTSwinBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoder

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

D_MODEL = 256
D_FFN = 2048
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4
N_ENC_LAYERS = 6
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


def load_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in sd.items()}


def profile_msdeform_attn_layer(attn, query_tt, src_tt, reference_points,
                                 spatial_shapes, level_start_index,
                                 padding_mask, normalizer,
                                 cpu_ref_levels, cpu_norm_levels,
                                 device):
    """Profile one MSDeformAttn forward call with per-op timing."""
    timings = {}

    M = attn.n_heads
    D = attn.d_per_head
    L = attn.n_levels
    P = attn.n_points

    # --- Linear projections ---
    ttnn.synchronize_device(device)
    t = time.time()
    value_tt = ttnn.linear(src_tt, attn.value_proj_w, bias=attn.value_proj_b)
    ttnn.synchronize_device(device)
    timings["linear_value_proj"] = time.time() - t

    t = time.time()
    offsets_tt = ttnn.linear(query_tt, attn.sampling_offsets_w, bias=attn.sampling_offsets_b)
    ttnn.synchronize_device(device)
    timings["linear_sampling_offsets"] = time.time() - t

    t = time.time()
    attn_w_tt = ttnn.linear(query_tt, attn.attention_weights_w, bias=attn.attention_weights_b)
    ttnn.synchronize_device(device)
    timings["linear_attn_weights"] = time.time() - t

    # --- Padding mask ---
    if padding_mask is not None:
        t = time.time()
        inv_mask = (~padding_mask).float().unsqueeze(-1).to(torch.bfloat16)
        inv_mask_tt = ttnn.from_torch(
            inv_mask, layout=ttnn.TILE_LAYOUT,
            device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        value_tt = ttnn.multiply(value_tt, inv_mask_tt)
        ttnn.deallocate(inv_mask_tt)
        ttnn.synchronize_device(device)
        timings["padding_mask"] = time.time() - t

    # --- Attention weights reshape + softmax ---
    t = time.time()
    attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.synchronize_device(device)
    timings["attn_w_to_rm"] = time.time() - t

    s = list(attn_w_tt.shape)
    N, Lq = s[0], s[1]

    t = time.time()
    attn_w_tt = ttnn.reshape(attn_w_tt, (1, 1, N * Lq * M, L * P))
    ttnn.synchronize_device(device)
    timings["attn_w_reshape1"] = time.time() - t

    t = time.time()
    attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.TILE_LAYOUT)
    ttnn.synchronize_device(device)
    timings["attn_w_to_tile_for_softmax"] = time.time() - t

    t = time.time()
    attn_w_tt = ttnn.softmax(attn_w_tt, dim=-1)
    ttnn.synchronize_device(device)
    timings["attn_w_softmax"] = time.time() - t

    t = time.time()
    attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
    attn_w_tt = ttnn.reshape(attn_w_tt, (N, Lq, M, L * P))
    attn_w_tt = ttnn.transpose(attn_w_tt, 1, 2)
    ttnn.synchronize_device(device)
    timings["attn_w_reshape2_transpose"] = time.time() - t

    t = time.time()
    attn_w_tt = ttnn.reshape(attn_w_tt, (N * M, Lq, L * P, 1))
    ttnn.synchronize_device(device)
    timings["attn_w_reshape3"] = time.time() - t

    # --- Value reshape ---
    t = time.time()
    value_tt = ttnn.to_layout(value_tt, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.synchronize_device(device)
    timings["value_to_rm"] = time.time() - t

    Len_in = list(value_tt.shape)[1]

    t = time.time()
    value_tt = ttnn.reshape(value_tt, (N, Len_in, M, D))
    ttnn.synchronize_device(device)
    timings["value_reshape"] = time.time() - t

    # --- Offsets to_torch (HOST SYNC) ---
    t = time.time()
    offsets_t = ttnn.to_torch(offsets_tt).float()
    timings["offsets_to_torch"] = time.time() - t
    ttnn.deallocate(offsets_tt)

    # --- CPU: reshape offsets for per-level fused grid ---
    t = time.time()
    offsets_t = offsets_t.view(N, Lq, M, L, P, 2)
    timings["cpu_offsets_reshape"] = time.time() - t

    # --- Per-level: fused grid precompute + grid_sample + attention ---
    level_timings = defaultdict(float)
    result_accum = None

    for lid, (H, W) in enumerate(spatial_shapes):
        H, W = int(H), int(W)
        start = int(level_start_index[lid])
        end = start + H * W

        # Value slice + reshape for this level
        ttnn.synchronize_device(device)
        t = time.time()
        val = ttnn.slice(value_tt, [0, start, 0, 0], [N, end, M, D])
        val = ttnn.transpose(val, 1, 2)
        val = ttnn.reshape(val, (N * M, H, W, D))
        ttnn.synchronize_device(device)
        level_timings["value_slice_reshape"] += time.time() - t

        # Fused grid precompute (sampling_locs + bilinear in one pass)
        t = time.time()
        off_l = offsets_t[:, :, :, lid, :, :]
        precomputed_tt = attn._fused_grid_for_level(
            off_l, cpu_ref_levels[lid], cpu_norm_levels[lid], H, W, N * M, Lq, P)
        level_timings["fused_grid_precompute"] += time.time() - t

        # grid_sample
        ttnn.synchronize_device(device)
        t = time.time()
        sampled = ttnn.grid_sample(
            val, precomputed_tt, use_precomputed_grid=True, align_corners=False)
        ttnn.synchronize_device(device)
        level_timings["grid_sample"] += time.time() - t
        ttnn.deallocate(val)
        ttnn.deallocate(precomputed_tt)

        # Attention weighting
        t = time.time()
        attn_l_tt = ttnn.slice(
            attn_w_tt, [0, 0, lid * P, 0], [N * M, Lq, (lid + 1) * P, 1])
        weighted = ttnn.multiply(attn_l_tt, sampled)
        ttnn.deallocate(attn_l_tt)
        ttnn.deallocate(sampled)
        level_sum = ttnn.sum(weighted, dim=2)
        ttnn.deallocate(weighted)
        ttnn.synchronize_device(device)
        level_timings["attn_weight_aggregate"] += time.time() - t

        if result_accum is None:
            result_accum = level_sum
        else:
            result_accum = ttnn.add(result_accum, level_sum)
            ttnn.deallocate(level_sum)

    for k, v in level_timings.items():
        timings[f"levels_{k}"] = v

    ttnn.deallocate(value_tt)
    ttnn.deallocate(attn_w_tt)

    # --- Result reshape + output_proj ---
    ttnn.synchronize_device(device)
    t = time.time()
    result = ttnn.to_layout(result_accum, ttnn.ROW_MAJOR_LAYOUT)
    result = ttnn.reshape(result, (N, M, Lq, D))
    result = ttnn.transpose(result, 1, 2)
    ttnn.synchronize_device(device)
    timings["result_reshape_transpose"] = time.time() - t

    t = time.time()
    result = ttnn.reshape(result, (N, Lq, M * D))
    result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
    ttnn.synchronize_device(device)
    timings["result_final_reshape"] = time.time() - t

    t = time.time()
    output = ttnn.linear(result, attn.output_proj_w, bias=attn.output_proj_b)
    ttnn.synchronize_device(device)
    timings["linear_output_proj"] = time.time() - t
    ttnn.deallocate(result)

    return output, timings


def profile_encoder_layer(layer, src, pos, reference_points, spatial_shapes,
                          level_start_index, key_padding_mask, normalizer,
                          cpu_ref_levels, cpu_norm_levels, device):
    """Profile one encoder layer (self_attn + FFN)."""
    timings = {}

    # query = src + pos
    ttnn.synchronize_device(device)
    t = time.time()
    query = ttnn.add(src, pos)
    ttnn.synchronize_device(device)
    timings["add_src_pos"] = time.time() - t

    # Self-attention (MSDeformAttn) - profiled in detail
    src2, attn_timings = profile_msdeform_attn_layer(
        layer.self_attn, query, src, reference_points, spatial_shapes,
        level_start_index, key_padding_mask, normalizer,
        cpu_ref_levels, cpu_norm_levels, device)
    ttnn.deallocate(query)
    timings.update(attn_timings)

    # Residual + LayerNorm1
    ttnn.synchronize_device(device)
    t = time.time()
    src = ttnn.add(src, src2)
    ttnn.deallocate(src2)
    src = ttnn.layer_norm(src, weight=layer.norm1_w, bias=layer.norm1_b)
    ttnn.synchronize_device(device)
    timings["residual_norm1"] = time.time() - t

    # FFN
    t = time.time()
    ffn = ttnn.linear(src, layer.ffn1_w, bias=layer.ffn1_b)
    ttnn.synchronize_device(device)
    timings["ffn_linear1"] = time.time() - t

    t = time.time()
    ffn = ttnn.relu(ffn)
    ttnn.synchronize_device(device)
    timings["ffn_relu"] = time.time() - t

    t = time.time()
    ffn = ttnn.linear(ffn, layer.ffn2_w, bias=layer.ffn2_b)
    ttnn.synchronize_device(device)
    timings["ffn_linear2"] = time.time() - t

    # Residual + LayerNorm2
    t = time.time()
    src = ttnn.add(src, ffn)
    ttnn.deallocate(ffn)
    src = ttnn.layer_norm(src, weight=layer.norm2_w, bias=layer.norm2_b)
    ttnn.synchronize_device(device)
    timings["residual_norm2"] = time.time() - t

    return src, timings


def main():
    device = ttnn.open_device(device_id=0)
    full_sd = load_state_dict()

    print("Building pipeline...")
    t0 = time.time()
    backbone = TTSwinBackbone(device, CHECKPOINT_PATH, use_compile=True)
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
    print(f"Built in {time.time() - t0:.1f}s\n")

    # Preprocess image
    tensor, mask = preprocess_image(IMAGE_PATH)
    print(f"Image: {os.path.basename(IMAGE_PATH)}, padded: {tensor.shape[2]}x{tensor.shape[3]}\n")

    # Run backbone (warm up + get features)
    with torch.no_grad():
        _ = backbone(tensor, mask)  # warm up
        bb_out = backbone(tensor, mask)

    # Upload to device
    src_tt = ttnn.from_torch(
        bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(
        bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    spatial_shapes = bb_out["spatial_shapes"]
    reference_points = bb_out["reference_points"]
    level_start_index = bb_out["level_start_index"]
    mask_flatten = bb_out["mask_flatten"]

    normalizer = torch.stack(
        [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1
    ).float()
    L = len(spatial_shapes)
    ref_f = reference_points.float()
    cpu_ref_levels = [ref_f[:, :, lid, :].unsqueeze(2).unsqueeze(3) for lid in range(L)]
    cpu_norm_levels = [normalizer[lid].view(1, 1, 1, 1, 2) for lid in range(L)]

    print(f"src shape: {list(src_tt.shape)}")
    print(f"spatial_shapes: {spatial_shapes}")
    print(f"level_start_index: {level_start_index}\n")

    # Warm-up run
    print("Warm-up run...")
    output = src_tt
    for layer in encoder.layers:
        output = layer(
            output, pos_tt, reference_points, spatial_shapes, level_start_index,
            mask_flatten, normalizer=normalizer,
            cpu_ref_levels=cpu_ref_levels, cpu_norm_levels=cpu_norm_levels)
    ttnn.synchronize_device(device)
    ttnn.deallocate(output)
    print("Warm-up done.\n")

    # Re-upload src (consumed by warm-up)
    src_tt = ttnn.from_torch(
        bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Profile each layer
    print("=" * 70)
    print("PROFILING ENCODER (6 layers)")
    print("=" * 70)

    all_layer_timings = []
    output = src_tt
    for i, layer in enumerate(encoder.layers):
        output, timings = profile_encoder_layer(
            layer, output, pos_tt, reference_points, spatial_shapes,
            level_start_index, mask_flatten, normalizer,
            cpu_ref_levels, cpu_norm_levels, device)
        all_layer_timings.append(timings)

        total = sum(timings.values())
        print(f"\nLayer {i}: {total*1000:.0f}ms")
        for k, v in sorted(timings.items(), key=lambda x: -x[1]):
            pct = v / total * 100 if total > 0 else 0
            print(f"  {k:>35}: {v*1000:>7.1f}ms  ({pct:>5.1f}%)")

    # Aggregate across all layers
    print(f"\n{'=' * 70}")
    print("AGGREGATE ACROSS ALL 6 LAYERS")
    print(f"{'=' * 70}")

    agg = defaultdict(float)
    for timings in all_layer_timings:
        for k, v in timings.items():
            agg[k] += v

    grand_total = sum(agg.values())
    print(f"\nTotal encoder time: {grand_total*1000:.0f}ms\n")

    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        pct = v / grand_total * 100 if grand_total > 0 else 0
        print(f"  {k:>35}: {v*1000:>7.1f}ms  ({pct:>5.1f}%)")

    # Group by category
    print(f"\n{'=' * 70}")
    print("GROUPED BY CATEGORY")
    print(f"{'=' * 70}")

    categories = {
        "Linear projections": ["linear_value_proj", "linear_sampling_offsets", "linear_attn_weights", "linear_output_proj"],
        "Attn weight pipeline": ["attn_w_to_rm", "attn_w_reshape1", "attn_w_to_tile_for_softmax",
                                  "attn_w_softmax", "attn_w_reshape2_transpose",
                                  "attn_w_reshape3"],
        "Value reshape": ["value_to_rm", "value_reshape"],
        "Offsets to_torch": ["offsets_to_torch"],
        "CPU grid compute": ["cpu_offsets_reshape", "levels_fused_grid_precompute"],
        "Device grid_sample": ["levels_grid_sample"],
        "Device value slice": ["levels_value_slice_reshape"],
        "Device attn aggregate": ["levels_attn_weight_aggregate"],
        "Result reshape": ["result_reshape_transpose", "result_final_reshape"],
        "FFN": ["ffn_linear1", "ffn_relu", "ffn_linear2"],
        "Residual+Norm": ["residual_norm1", "residual_norm2"],
        "Other": ["add_src_pos", "padding_mask"],
    }

    for cat_name, keys in categories.items():
        cat_total = sum(agg.get(k, 0) for k in keys)
        pct = cat_total / grand_total * 100 if grand_total > 0 else 0
        print(f"  {cat_name:>25}: {cat_total*1000:>7.0f}ms  ({pct:>5.1f}%)")

    ttnn.deallocate(output)
    ttnn.deallocate(pos_tt)
    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
