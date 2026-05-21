# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile encoder MSDeformAttn sub-op timing at 80997 queries scale."""

import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_encoder import TTDeformableEncoder

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")
D_MODEL, D_FFN, N_HEADS, N_LEVELS, N_POINTS = 256, 1024, 8, 5, 4
N_ENC_LAYERS = 6


def load_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in sd.items()}


def profile_encoder_msdeform(device, cross_attn, query_tt, input_tt,
                              reference_points, spatial_shapes,
                              level_start_index, padding_mask, tag):
    records = {}

    def sync():
        ttnn.synchronize_device(device)

    def timed(name, fn):
        sync()
        t0 = time.perf_counter()
        r = fn()
        sync()
        records[name] = time.perf_counter() - t0
        return r

    M = cross_attn.n_heads
    D = cross_attn.d_per_head
    L = cross_attn.n_levels
    P = cross_attn.n_points

    value_tt = timed(f"{tag}/linear_value",
        lambda: ttnn.linear(input_tt, cross_attn.value_proj_w, bias=cross_attn.value_proj_b))
    offsets_tt = timed(f"{tag}/linear_offsets",
        lambda: ttnn.linear(query_tt, cross_attn.sampling_offsets_w, bias=cross_attn.sampling_offsets_b))
    attn_w_tt = timed(f"{tag}/linear_attn_w",
        lambda: ttnn.linear(query_tt, cross_attn.attention_weights_w, bias=cross_attn.attention_weights_b))

    def apply_mask():
        nonlocal value_tt
        if padding_mask is not None:
            inv_mask = (~padding_mask).float().unsqueeze(-1).to(torch.bfloat16)
            inv_mask_tt = ttnn.from_torch(inv_mask, layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            value_tt = ttnn.multiply(value_tt, inv_mask_tt)
            ttnn.deallocate(inv_mask_tt)
    timed(f"{tag}/device_value_mask", apply_mask)

    offsets_t = timed(f"{tag}/to_torch_offsets", lambda: ttnn.to_torch(offsets_tt).float())
    ttnn.deallocate(offsets_tt)

    N = offsets_t.shape[0]
    Lq = offsets_t.shape[1]
    Len_in = sum(int(H) * int(W) for H, W in spatial_shapes)

    def device_attn_softmax():
        nonlocal attn_w_tt
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
        attn_w_tt = ttnn.reshape(attn_w_tt, (1, 1, N * Lq * M, L * P))
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.TILE_LAYOUT)
        attn_w_tt = ttnn.softmax(attn_w_tt, dim=-1)
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
        attn_w_tt = ttnn.reshape(attn_w_tt, (N, Lq, M, L * P))
        attn_w_tt = ttnn.transpose(attn_w_tt, 1, 2)
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.TILE_LAYOUT)
        attn_w_tt = ttnn.to_layout(attn_w_tt, ttnn.ROW_MAJOR_LAYOUT)
        attn_w_tt = ttnn.reshape(attn_w_tt, (N * M, Lq, L * P, 1))
    timed(f"{tag}/device_attn_softmax", device_attn_softmax)

    def value_reshape():
        nonlocal value_tt
        value_tt = ttnn.to_layout(value_tt, ttnn.ROW_MAJOR_LAYOUT)
        value_tt = ttnn.reshape(value_tt, (N, Len_in, M, D))
    timed(f"{tag}/device_value_reshape", value_reshape)

    def host_offsets_reshape():
        nonlocal offsets_t
        offsets_t = offsets_t.view(N, Lq, M, L, P, 2)
    timed(f"{tag}/host_offsets_reshape", host_offsets_reshape)

    def compute_sampling_locs():
        ref = reference_points.float()
        normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).float()
        if ref.shape[-1] == 2:
            return ref[:, :, None, :, None, :] + offsets_t / normalizer[None, None, None, :, None, :]
        else:
            return ref[:, :, None, :, None, :2] + offsets_t / P * ref[:, :, None, :, None, 2:] * 0.5
    sampling_locs = timed(f"{tag}/host_sampling_locs", compute_sampling_locs)

    def compute_grids():
        return 2 * sampling_locs - 1
    sampling_grids = timed(f"{tag}/host_sampling_grids", compute_grids)

    result_accum = None
    for lid, (H, W) in enumerate(spatial_shapes):
        H, W = int(H), int(W)
        start = int(level_start_index[lid])
        end = start + H * W

        def device_val_prep(start=start, end=end, H=H, W=W):
            val = ttnn.slice(value_tt, [0, start, 0, 0], [N, end, M, D])
            val = ttnn.transpose(val, 1, 2)
            val = ttnn.to_layout(val, ttnn.TILE_LAYOUT)
            val = ttnn.to_layout(val, ttnn.ROW_MAJOR_LAYOUT)
            val = ttnn.reshape(val, (N * M, H, W, D))
            return val
        val = timed(f"{tag}/device_val_prep_L{lid}({H}x{W})", device_val_prep)

        def host_grid_prep(lid=lid):
            return (sampling_grids[:, :, :, lid].transpose(1, 2)
                .flatten(0, 1).contiguous().float())
        grid_l = timed(f"{tag}/host_grid_prep_L{lid}", host_grid_prep)

        def precompute_grid(grid_l=grid_l, H=H, W=W):
            grid_host = ttnn.from_torch(grid_l, dtype=ttnn.float32)
            precomputed = ttnn.prepare_grid_sample_grid(
                grid_host, [N * M, H, W, D],
                mode="bilinear", padding_mode="zeros", align_corners=False,
                output_dtype=ttnn.bfloat16)
            precomputed_t = ttnn.to_torch(precomputed)
            precomputed_packed = precomputed_t.reshape(N * M, Lq, 1, 6 * P)
            return ttnn.from_torch(precomputed_packed.contiguous(),
                layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        precomputed_tt = timed(f"{tag}/precompute_grid_L{lid}", precompute_grid)

        def do_grid_sample(val=val, precomputed_tt=precomputed_tt):
            return ttnn.grid_sample(val, precomputed_tt,
                use_precomputed_grid=True, align_corners=False)
        sampled = timed(f"{tag}/grid_sample_L{lid}({H}x{W})", do_grid_sample)
        ttnn.deallocate(val)
        ttnn.deallocate(precomputed_tt)

        def device_attn_slice(lid=lid):
            return ttnn.slice(attn_w_tt, [0, 0, lid * P, 0], [N * M, Lq, (lid + 1) * P, 1])
        attn_l_tt = timed(f"{tag}/device_attn_slice_L{lid}", device_attn_slice)

        def device_weighted_sum(sampled=sampled, attn_l_tt=attn_l_tt):
            weighted = ttnn.multiply(attn_l_tt, sampled)
            ttnn.deallocate(attn_l_tt)
            ttnn.deallocate(sampled)
            level_sum = ttnn.sum(weighted, dim=2)
            ttnn.deallocate(weighted)
            return level_sum
        level_sum = timed(f"{tag}/device_weighted_sum_L{lid}", device_weighted_sum)

        if result_accum is None:
            result_accum = level_sum
        else:
            def device_accum(result_accum=result_accum, level_sum=level_sum):
                r = ttnn.add(result_accum, level_sum)
                ttnn.deallocate(level_sum)
                return r
            result_accum = timed(f"{tag}/device_accum_L{lid}", device_accum)

    ttnn.deallocate(value_tt)
    ttnn.deallocate(attn_w_tt)
    result = result_accum

    def device_merge_heads():
        nonlocal result
        result = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)
        result = ttnn.reshape(result, (N, M, Lq, D))
        result = ttnn.transpose(result, 1, 2)
        result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
        result = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT)
        result = ttnn.reshape(result, (N, Lq, M * D))
        result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
    timed(f"{tag}/device_merge_heads", device_merge_heads)

    def output_proj():
        out = ttnn.linear(result, cross_attn.output_proj_w, bias=cross_attn.output_proj_b)
        ttnn.deallocate(result)
        return out
    final = timed(f"{tag}/linear_output", output_proj)

    return final, records


def main():
    device = ttnn.open_device(device_id=0)
    print("Device opened.")
    full_sd = load_state_dict()

    print("Building backbone + encoder...")
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN,
                                   N_LEVELS, N_HEADS, N_POINTS)
    print("Pipeline built.")

    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    print("Running backbone...")
    with torch.no_grad():
        bb_out = backbone(image_tensor, mask)
    print(f"Backbone done. Tokens: {bb_out['src_flatten'].shape[1]}")
    print(f"Spatial shapes: {bb_out['spatial_shapes'].tolist()}")

    src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Warm-up: run full encoder once
    print("\nWarm-up: running full encoder...")
    t0 = time.time()
    with torch.no_grad():
        enc_out = encoder(src_tt, pos_tt, bb_out["reference_points"],
                          bb_out["spatial_shapes"], bb_out["level_start_index"],
                          bb_out["mask_flatten"])
    print(f"Warm-up encoder: {(time.time()-t0)*1000:.0f}ms")
    ttnn.deallocate(enc_out)

    # Profile 2 encoder layers (L0 cold-ish, L1 warm)
    print("\n--- Profiling encoder MSDeformAttn (layer-by-layer) ---")
    N_PROFILE_LAYERS = 2
    all_records = {}

    current_src = src_tt
    for layer_id in range(N_PROFILE_LAYERS):
        layer = encoder.layers[layer_id]
        query = ttnn.add(current_src, pos_tt)

        result, recs = profile_encoder_msdeform(
            device, layer.self_attn,
            query, current_src,
            bb_out["reference_points"],
            bb_out["spatial_shapes"],
            bb_out["level_start_index"],
            bb_out["mask_flatten"],
            f"EncL{layer_id}(Lq=80997)")
        all_records.update(recs)
        ttnn.deallocate(query)

        # Complete the encoder layer (norm + FFN) to get input for next layer
        current_src = ttnn.add(current_src, result)
        ttnn.deallocate(result)
        current_src = ttnn.layer_norm(current_src, weight=layer.norm1_w, bias=layer.norm1_b)
        ffn = ttnn.linear(current_src, layer.ffn1_w, bias=layer.ffn1_b)
        ffn = ttnn.relu(ffn)
        ffn = ttnn.linear(ffn, layer.ffn2_w, bias=layer.ffn2_b)
        current_src = ttnn.add(current_src, ffn)
        ttnn.deallocate(ffn)
        current_src = ttnn.layer_norm(current_src, weight=layer.norm2_w, bias=layer.norm2_b)

    ttnn.deallocate(current_src)
    ttnn.deallocate(src_tt)
    ttnn.deallocate(pos_tt)

    # Print detailed results
    print(f"\n{'='*80}")
    print(f"  Encoder MSDeformAttn per-op profiling ({N_PROFILE_LAYERS} layers)")
    print(f"{'='*80}")
    total = sum(all_records.values())
    for k, v in sorted(all_records.items(), key=lambda x: -x[1]):
        print(f"  {k:>55s}: {v*1000:>8.2f}ms  ({v/total*100:>5.1f}%)")
    print(f"\n  {'TOTAL':>55s}: {total*1000:>8.2f}ms")

    # Category summary
    cats = defaultdict(float)
    for k, v in all_records.items():
        parts = k.split('/')
        cat = parts[1]
        for prefix in ["grid_sample_L", "device_val_prep_L", "host_grid_prep_L", "precompute_grid_L"]:
            if cat.startswith(prefix):
                cat = prefix.rstrip("_L")
                break
        cats[cat] += v

    print(f"\n{'='*80}")
    print(f"  Encoder MSDeformAttn by category (summed across {N_PROFILE_LAYERS} layers)")
    print(f"{'='*80}")
    for cat, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:>35s}: {v*1000:>8.2f}ms  ({v/total*100:>5.1f}%)")
    print(f"  {'TOTAL':>35s}: {total*1000:>8.2f}ms")
    print(f"  {'Per layer avg':>35s}: {total/N_PROFILE_LAYERS*1000:>8.2f}ms")
    print(f"  {'Estimated 6 layers':>35s}: {total/N_PROFILE_LAYERS*6*1000:>8.2f}ms")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
