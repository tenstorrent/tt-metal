# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile the new MSDeformAttn (Phase 1+2) sub-op timing."""

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
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import TwoStageQueryGenerator
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import (
    TTEDPoseDecoder, inverse_sigmoid, gen_sineembed_for_position,
)

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")
D_MODEL, D_FFN, N_HEADS, N_LEVELS, N_POINTS = 256, 2048, 8, 5, 4
N_ENC_LAYERS = N_DEC_LAYERS = 6
NUM_QUERIES, NUM_CLASSES, NUM_BODY_POINTS, NUM_BOX_DEC_LAYERS, NUM_GROUP = 900, 2, 17, 2, 100


def load_state_dict():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_ema", ckpt.get("model", ckpt))
    return {k.replace("module.", ""): v for k, v in sd.items()}


def preprocess_image(image_path, max_size=1333, target_size=800):
    from PIL import Image
    from torchvision import transforms
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = target_size / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > max_size:
        scale = max_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    normalize = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tensor = normalize(img)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    mask = torch.ones(tensor.shape[1], tensor.shape[2], dtype=torch.bool)
    mask[:new_h, :new_w] = False
    return tensor.unsqueeze(0), mask.unsqueeze(0), torch.tensor([[orig_h, orig_w]])


def profile_new_msdeform(device, cross_attn, query_bf, ref_pts, memory_tt,
                         spatial_shapes, level_start_index, padding_mask, tag):
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

    query_tt = timed(f"{tag}/ensure_query", lambda: cross_attn._ensure_tt(query_bf))

    value_tt = timed(f"{tag}/linear_value",
        lambda: ttnn.linear(memory_tt, cross_attn.value_proj_w, bias=cross_attn.value_proj_b))
    offsets_tt = timed(f"{tag}/linear_offsets",
        lambda: ttnn.linear(query_tt, cross_attn.sampling_offsets_w, bias=cross_attn.sampling_offsets_b))
    attn_w_tt = timed(f"{tag}/linear_attn_w",
        lambda: ttnn.linear(query_tt, cross_attn.attention_weights_w, bias=cross_attn.attention_weights_b))

    # Value mask on device
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
    attn_w_t = timed(f"{tag}/to_torch_attn_w", lambda: ttnn.to_torch(attn_w_tt).float())
    ttnn.deallocate(offsets_tt); ttnn.deallocate(attn_w_tt)

    N = offsets_t.shape[0]
    Lq = offsets_t.shape[1]
    Len_in = sum(int(H) * int(W) for H, W in spatial_shapes)

    # Value reshape on device
    def value_reshape():
        nonlocal value_tt
        value_tt = ttnn.to_layout(value_tt, ttnn.ROW_MAJOR_LAYOUT)
        value_tt = ttnn.reshape(value_tt, (N, Len_in, M, D))
    timed(f"{tag}/device_value_reshape", value_reshape)

    # Host: offsets reshape + softmax + sampling_locs
    def host_offsets_softmax():
        nonlocal offsets_t, attn_w_t
        offsets_t = offsets_t.view(N, Lq, M, L, P, 2)
        attn_w_t = F.softmax(attn_w_t.view(N, Lq, M, L * P), dim=-1).view(N, Lq, M, L, P)
    timed(f"{tag}/host_offsets_softmax", host_offsets_softmax)

    def compute_sampling_locs():
        ref = ref_pts.float()
        normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1).float()
        if ref.shape[-1] == 2:
            return ref[:, :, None, :, None, :] + offsets_t / normalizer[None, None, None, :, None, :]
        else:
            return ref[:, :, None, :, None, :2] + offsets_t / P * ref[:, :, None, :, None, 2:] * 0.5
    sampling_locs = timed(f"{tag}/host_sampling_locs", compute_sampling_locs)
    sampling_grids = 2 * sampling_locs - 1

    # Per-level
    sampled_levels = []
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
        out = timed(f"{tag}/grid_sample_L{lid}({H}x{W})", do_grid_sample)
        sampled_levels.append(out)
        ttnn.deallocate(val)
        ttnn.deallocate(precomputed_tt)

    ttnn.deallocate(value_tt)

    # Device aggregate
    def device_concat():
        return ttnn.concat(sampled_levels, dim=2)
    all_sampled = timed(f"{tag}/device_concat", device_concat)
    for s in sampled_levels:
        ttnn.deallocate(s)

    def host_attn_prep():
        attn_agg = (attn_w_t.transpose(1, 2).reshape(N * M, Lq, L * P)
            .unsqueeze(2).contiguous().to(torch.bfloat16))
        return ttnn.from_torch(attn_agg, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    attn_agg_tt = timed(f"{tag}/host_attn_to_device", host_attn_prep)

    def device_matmul():
        nonlocal all_sampled, attn_agg_tt
        all_sampled = ttnn.to_layout(all_sampled, ttnn.TILE_LAYOUT)
        attn_agg_tt = ttnn.to_layout(attn_agg_tt, ttnn.TILE_LAYOUT)
        return ttnn.matmul(attn_agg_tt, all_sampled)
    result = timed(f"{tag}/device_aggregate_matmul", device_matmul)
    ttnn.deallocate(all_sampled)
    ttnn.deallocate(attn_agg_tt)

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
    import json
    device = ttnn.open_device(device_id=0)
    print("Device opened.")
    full_sd = load_state_dict()

    print("Building pipeline...")
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    decoder = TTEDPoseDecoder(device, full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
                              N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES, NUM_BODY_POINTS, NUM_BOX_DEC_LAYERS, NUM_GROUP)
    print("Pipeline built.")

    coco_dir = "/home/yito/datasets/coco"
    ann_file = os.path.join(coco_dir, "annotations", "person_keypoints_val2017.json")
    with open(ann_file) as f:
        ann = json.load(f)
    img_path = os.path.join(coco_dir, "val2017", ann["images"][0]["file_name"])

    tensor, mask, orig_size = preprocess_image(img_path)
    with torch.no_grad():
        bb_out = backbone(tensor, mask)
    src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    with torch.no_grad():
        enc_out_tt = encoder(src_tt, pos_tt, bb_out["reference_points"], bb_out["spatial_shapes"],
                             bb_out["level_start_index"], bb_out["mask_flatten"])
    memory = ttnn.to_torch(enc_out_tt).float()
    ttnn.deallocate(enc_out_tt); ttnn.deallocate(src_tt); ttnn.deallocate(pos_tt)
    with torch.no_grad():
        query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
    memory_tt = ttnn.from_torch(memory.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    print("Encoder + two-stage done.\n")

    # Warm up
    print("--- Warm-up ---")
    attn_mask2 = torch.zeros(1, N_HEADS, NUM_GROUP*(NUM_BODY_POINTS+1), NUM_GROUP*(NUM_BODY_POINTS+1), dtype=torch.bool)
    group_size = NUM_BODY_POINTS + 1
    total_q = NUM_GROUP * group_size
    kpt_index = [x for x in range(total_q) if x % group_size == 0]
    for matchj in range(total_q):
        sj = (matchj // group_size) * group_size
        ej = (matchj // group_size + 1) * group_size
        if sj > 0: attn_mask2[:, :, matchj, :sj] = True
        if ej < total_q: attn_mask2[:, :, matchj, ej:] = True
    for match_x in range(total_q):
        if match_x % group_size == 0:
            attn_mask2[:, :, match_x, kpt_index] = False
    float_mask2 = torch.zeros_like(attn_mask2, dtype=torch.bfloat16)
    float_mask2.masked_fill_(attn_mask2, float("-inf"))
    attn_mask2_tt = ttnn.from_torch(float_mask2, layout=ttnn.TILE_LAYOUT, device=device,
                                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    with torch.no_grad():
        decoder(tgt=query_out["tgt"], memory_tt=memory_tt,
                refpoint_embed=query_out["refpoint_embed"],
                spatial_shapes=bb_out["spatial_shapes"],
                level_start_index=bb_out["level_start_index"],
                valid_ratios=bb_out["valid_ratios"],
                memory_key_padding_mask=bb_out["mask_flatten"],
                self_attn_mask=None, self_attn_mask2=attn_mask2_tt)
    print("Warm-up done.\n")

    # Profile
    print("--- Profiling cross-attention (new path) ---")
    output = query_out["tgt"].transpose(0, 1)
    reference_points = query_out["refpoint_embed"].transpose(0, 1).sigmoid().clamp(1e-3, 1 - 1e-3)
    valid_ratios = bb_out["valid_ratios"]
    spatial_shapes = bb_out["spatial_shapes"]
    level_start_index = bb_out["level_start_index"]

    all_records = {}
    with torch.no_grad():
        for layer_id in range(N_DEC_LAYERS):
            if reference_points.shape[-1] == 4:
                ref_input = reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            else:
                ref_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine = gen_sineembed_for_position(ref_input[:, :, 0, :])
            query_pos = decoder.ref_point_head(query_sine)

            tgt_mask = None if layer_id < NUM_BOX_DEC_LAYERS else attn_mask2_tt
            tgt_bf = output.to(torch.bfloat16).transpose(0, 1).contiguous()
            qk_input = output + query_pos
            qk_bf = qk_input.to(torch.bfloat16).transpose(0, 1).contiguous()
            sa_out = decoder.layers[layer_id]._self_attention_device(tgt_bf, qk_bf, tgt_mask)
            tgt_resid = output.float().transpose(0, 1).contiguous() + sa_out
            tgt_tt = ttnn.from_torch(tgt_resid.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tgt_tt = ttnn.layer_norm(tgt_tt, weight=decoder.layers[layer_id].norm2_w, bias=decoder.layers[layer_id].norm2_b)
            output_normed = ttnn.to_torch(tgt_tt).transpose(0, 1).contiguous().float()
            ttnn.deallocate(tgt_tt)

            query_with_pos = output_normed + query_pos
            query_bf = query_with_pos.transpose(0, 1).contiguous().to(torch.bfloat16)
            ref_pts_bf = ref_input.transpose(0, 1).contiguous()

            Lq = query_bf.shape[1]
            result, recs = profile_new_msdeform(
                device, decoder.layers[layer_id].cross_attn,
                query_bf, ref_pts_bf, memory_tt,
                spatial_shapes, level_start_index,
                bb_out["mask_flatten"], f"L{layer_id}(Lq={Lq})")
            all_records.update(recs)
            cross_out = ttnn.to_torch(result).transpose(0, 1).contiguous().float()
            ttnn.deallocate(result)

            output = output_normed + cross_out
            tgt_tt = ttnn.from_torch(output.to(torch.bfloat16).transpose(0, 1).contiguous(),
                layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tgt_tt = ttnn.layer_norm(tgt_tt, weight=decoder.layers[layer_id].norm1_w, bias=decoder.layers[layer_id].norm1_b)
            layer_obj = decoder.layers[layer_id]
            ffn = ttnn.linear(tgt_tt, layer_obj.ffn1_w, bias=layer_obj.ffn1_b)
            ffn = ttnn.relu(ffn)
            ffn = ttnn.linear(ffn, layer_obj.ffn2_w, bias=layer_obj.ffn2_b)
            tgt_tt = ttnn.add(tgt_tt, ffn)
            ttnn.deallocate(ffn)
            tgt_tt = ttnn.layer_norm(tgt_tt, weight=layer_obj.norm3_w, bias=layer_obj.norm3_b)
            output = ttnn.to_torch(tgt_tt).transpose(0, 1).contiguous().float()
            ttnn.deallocate(tgt_tt)

            if layer_id < NUM_BOX_DEC_LAYERS:
                ref_before = inverse_sigmoid(reference_points)
                delta = decoder.bbox_embed[layer_id](output)
                new_ref = (delta + ref_before).sigmoid().clamp(1e-3, 1 - 1e-3)
            if layer_id == NUM_BOX_DEC_LAYERS - 1:
                cls = decoder.class_embed[layer_id](output)
                topk = torch.topk(cls.max(-1)[0], NUM_GROUP, dim=0)[1]
                new_ref_box = torch.gather(new_ref, 0, topk.unsqueeze(-1).repeat(1, 1, 4))
                new_out_box = torch.gather(output, 0, topk.unsqueeze(-1).repeat(1, 1, D_MODEL))
                new_out_kpt = new_out_box[:, None, :, :] + decoder.keypoint_embed.weight[None, :, None, :]
                d_xy = decoder.pose_embed[-1](new_out_kpt)[..., :2]
                kpt_xy = (inverse_sigmoid(new_ref_box[..., :2][:, None]) + d_xy).sigmoid().clamp(1e-3, 1-1e-3)
                nqb, _, bs_, _ = kpt_xy.shape
                kw = decoder.hw.weight.unsqueeze(0).unsqueeze(-2).repeat(nqb, 1, bs_, 1).sigmoid()
                kpt_wh = kw * new_ref_box[..., 2:][:, None]
                kpt_ref = torch.cat((kpt_xy, kpt_wh), dim=-1)
                new_ref = torch.cat((new_ref_box.unsqueeze(1), kpt_ref), dim=1).flatten(0, 1)
                output = torch.cat((new_out_box.unsqueeze(1), new_out_kpt), dim=1).flatten(0, 1)
            if layer_id >= NUM_BOX_DEC_LAYERS:
                ref_before = inverse_sigmoid(reference_points)
                out_bbox = output[0::(NUM_BODY_POINTS+1)]
                ref_bbox = ref_before[0::(NUM_BODY_POINTS+1)]
                d_bbox = decoder.bbox_embed[layer_id](out_bbox)
                new_ref_box = (d_bbox + ref_bbox).sigmoid().clamp(1e-3, 1-1e-3)
                ki = decoder.kpt_index
                out_kpt = output.index_select(0, torch.tensor(ki))
                d_xy = decoder.pose_embed[layer_id - NUM_BOX_DEC_LAYERS](out_kpt)
                ref_kpt = ref_before.index_select(0, torch.tensor(ki)).clone()
                d_hw = decoder.pose_hw_embed[layer_id - NUM_BOX_DEC_LAYERS](out_kpt)
                ref_kpt[..., :2] += d_xy[..., :2]
                ref_kpt[..., 2:] += d_hw
                new_ref_kpt = ref_kpt.sigmoid().clamp(1e-3, 1-1e-3)
                bs_ = new_ref_box.shape[1]
                new_ref = torch.cat((new_ref_box.unsqueeze(1),
                    new_ref_kpt.view(-1, NUM_BODY_POINTS, bs_, 4)), dim=1).flatten(0, 1)
            reference_points = new_ref.detach()

    print(f"\n{'='*80}")
    print(f"  MSDeformAttn per-op profiling (new path, all 6 layers)")
    print(f"{'='*80}")
    total = sum(all_records.values())
    for k, v in sorted(all_records.items(), key=lambda x: -x[1]):
        print(f"  {k:>55s}: {v*1000:>8.2f}ms  ({v/total*100:>5.1f}%)")
    print(f"\n  {'TOTAL':>55s}: {total*1000:>8.2f}ms")

    cats = defaultdict(float)
    for k, v in all_records.items():
        parts = k.split('/')
        cat = parts[1]
        if cat.startswith("grid_sample_L"):
            cat = "grid_sample"
        elif cat.startswith("device_val_prep_L"):
            cat = "device_val_prep"
        elif cat.startswith("host_grid_prep_L"):
            cat = "host_grid_prep"
        elif cat.startswith("precompute_grid_L"):
            cat = "precompute_grid"
        cats[cat] += v
    print(f"\n{'='*80}")
    print(f"  MSDeformAttn by category (summed across layers)")
    print(f"{'='*80}")
    for cat, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:>35s}: {v*1000:>8.2f}ms  ({v/total*100:>5.1f}%)")

    ttnn.deallocate(memory_tt)
    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
