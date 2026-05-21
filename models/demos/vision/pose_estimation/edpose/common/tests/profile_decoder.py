# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Decoder profiling: measure per-op timing with device synchronization barriers.

Instruments TTDeformableDecoderLayer to measure:
  - Self-attention: QKV linear, split_heads, matmul, scale, mask, softmax, merge, out_proj
  - Cross-attention: MSDeformAttn total
  - FFN: linear1, relu, linear2, add, layer_norm
  - Host↔device transfers: from_torch, to_torch
"""

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
    TTEDPoseDecoder,
    MLP,
    inverse_sigmoid,
    gen_sineembed_for_position,
)
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_deformable_decoder import TTDeformableDecoderLayer

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

D_MODEL = 256
D_FFN = 2048
N_HEADS = 8
N_LEVELS = 5
N_POINTS = 4
N_ENC_LAYERS = 6
N_DEC_LAYERS = 6
NUM_QUERIES = 900
NUM_CLASSES = 2
NUM_BODY_POINTS = 17
NUM_BOX_DEC_LAYERS = 2
NUM_GROUP = 100
HEAD_DIM = D_MODEL // N_HEADS


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
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = normalize(img)
    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h))
    mask = torch.ones(tensor.shape[1], tensor.shape[2], dtype=torch.bool)
    mask[:new_h, :new_w] = False
    return tensor.unsqueeze(0), mask.unsqueeze(0), torch.tensor([[orig_h, orig_w]])


class Timer:
    def __init__(self, device):
        self.device = device
        self.records = defaultdict(list)

    def sync(self):
        ttnn.synchronize_device(self.device)

    def measure(self, name, fn):
        self.sync()
        t0 = time.perf_counter()
        result = fn()
        self.sync()
        dt = time.perf_counter() - t0
        self.records[name].append(dt)
        return result

    def report(self, title=""):
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
        total = 0
        for name, times in self.records.items():
            avg = sum(times) / len(times)
            total += avg * len(times)
            print(f"  {name:>40s}: {avg*1000:>8.2f}ms  (x{len(times)}  total={sum(times)*1000:.1f}ms)")
        print(f"  {'TOTAL':>40s}: {total*1000:>8.2f}ms")


def profile_decoder_layer(timer, layer, tgt, tgt_query_pos, tgt_reference_points,
                          memory, spatial_shapes, level_start_index,
                          memory_key_padding_mask, self_attn_mask, layer_id):
    """Profile a single decoder layer with fine-grained timing."""
    tag = f"L{layer_id}"

    if layer.has_self_attn:
        # Prepare inputs
        def prep_sa():
            tgt_bf = tgt.to(torch.bfloat16).transpose(0, 1).contiguous()
            qk_input = tgt + tgt_query_pos if tgt_query_pos is not None else tgt
            qk_bf = qk_input.to(torch.bfloat16).transpose(0, 1).contiguous()
            return tgt_bf, qk_bf
        tgt_bf, qk_bf = prep_sa()
        N, Lq, C = qk_bf.shape

        # --- Self-attention sub-ops ---
        qk_tt = timer.measure(f"{tag}/sa/from_torch(qk)", lambda: ttnn.from_torch(
            qk_bf, layout=ttnn.TILE_LAYOUT, device=layer.device, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        v_tt = timer.measure(f"{tag}/sa/from_torch(v)", lambda: ttnn.from_torch(
            tgt_bf, layout=ttnn.TILE_LAYOUT, device=layer.device, memory_config=ttnn.DRAM_MEMORY_CONFIG))

        q = timer.measure(f"{tag}/sa/linear_q", lambda: ttnn.linear(qk_tt, layer.sa_q_w, bias=layer.sa_q_b))
        k = timer.measure(f"{tag}/sa/linear_k", lambda: ttnn.linear(qk_tt, layer.sa_k_w, bias=layer.sa_k_b))
        v = timer.measure(f"{tag}/sa/linear_v", lambda: ttnn.linear(v_tt, layer.sa_v_w, bias=layer.sa_v_b))
        ttnn.deallocate(qk_tt); ttnn.deallocate(v_tt)

        q = timer.measure(f"{tag}/sa/split_heads_q", lambda: layer._split_heads(q, N, Lq))
        k = timer.measure(f"{tag}/sa/split_heads_k", lambda: layer._split_heads(k, N, Lq))
        v = timer.measure(f"{tag}/sa/split_heads_v", lambda: layer._split_heads(v, N, Lq))

        k_t = timer.measure(f"{tag}/sa/transpose_k", lambda: ttnn.transpose(k, -2, -1))
        ttnn.deallocate(k)
        attn = timer.measure(f"{tag}/sa/matmul_qk", lambda: ttnn.matmul(q, k_t))
        ttnn.deallocate(q); ttnn.deallocate(k_t)
        attn = timer.measure(f"{tag}/sa/scale", lambda: ttnn.multiply(attn, HEAD_DIM ** -0.5))

        if self_attn_mask is not None:
            attn = timer.measure(f"{tag}/sa/mask_add", lambda: ttnn.add(attn, self_attn_mask))

        attn = timer.measure(f"{tag}/sa/softmax", lambda: ttnn.softmax(attn, dim=-1))
        out = timer.measure(f"{tag}/sa/matmul_av", lambda: ttnn.matmul(attn, v))
        ttnn.deallocate(attn); ttnn.deallocate(v)

        out = timer.measure(f"{tag}/sa/merge_heads", lambda: layer._merge_heads(out, N, Lq))
        result = timer.measure(f"{tag}/sa/linear_out", lambda: ttnn.linear(out, layer.sa_out_w, bias=layer.sa_out_b))
        ttnn.deallocate(out)

        sa_out = timer.measure(f"{tag}/sa/to_torch", lambda: ttnn.to_torch(result).float())
        ttnn.deallocate(result)

        # Residual + norm2
        def do_resid_norm2():
            tgt_resid = tgt.float().transpose(0, 1).contiguous() + sa_out
            tgt_tt = ttnn.from_torch(tgt_resid.to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT, device=layer.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            tgt_tt = ttnn.layer_norm(tgt_tt, weight=layer.norm2_w, bias=layer.norm2_b)
            t = ttnn.to_torch(tgt_tt).transpose(0, 1).contiguous().float()
            ttnn.deallocate(tgt_tt)
            return t
        tgt_new = timer.measure(f"{tag}/resid+norm2", do_resid_norm2)
    else:
        tgt_new = tgt

    # --- Cross-attention ---
    def do_cross_attn():
        query_with_pos = tgt_new + tgt_query_pos if tgt_query_pos is not None else tgt_new
        query_bf = query_with_pos.transpose(0, 1).contiguous()
        ref_pts_bf = tgt_reference_points.transpose(0, 1).contiguous()
        cross_out = layer.cross_attn(
            query_bf.to(torch.bfloat16), ref_pts_bf, memory,
            spatial_shapes, level_start_index, memory_key_padding_mask)
        cross_out_torch = ttnn.to_torch(cross_out).transpose(0, 1).contiguous().float()
        ttnn.deallocate(cross_out)
        return cross_out_torch
    cross_out_torch = timer.measure(f"{tag}/cross_attn", do_cross_attn)

    # Residual + norm1 + FFN + norm3
    def do_ffn_block():
        t = tgt_new + cross_out_torch
        tgt_tt = ttnn.from_torch(t.to(torch.bfloat16).transpose(0, 1).contiguous(),
            layout=ttnn.TILE_LAYOUT, device=layer.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tgt_tt = ttnn.layer_norm(tgt_tt, weight=layer.norm1_w, bias=layer.norm1_b)
        ffn = ttnn.linear(tgt_tt, layer.ffn1_w, bias=layer.ffn1_b)
        ffn = ttnn.relu(ffn)
        ffn = ttnn.linear(ffn, layer.ffn2_w, bias=layer.ffn2_b)
        tgt_tt = ttnn.add(tgt_tt, ffn)
        ttnn.deallocate(ffn)
        tgt_tt = ttnn.layer_norm(tgt_tt, weight=layer.norm3_w, bias=layer.norm3_b)
        r = ttnn.to_torch(tgt_tt).transpose(0, 1).contiguous().float()
        ttnn.deallocate(tgt_tt)
        return r
    result = timer.measure(f"{tag}/norm1+FFN+norm3", do_ffn_block)
    return result


def prepare_attn_mask2(device):
    total_q = NUM_GROUP * (NUM_BODY_POINTS + 1)
    group_size = NUM_BODY_POINTS + 1
    kpt_index = [x for x in range(total_q) if x % group_size == 0]
    attn_mask = torch.zeros(1, N_HEADS, total_q, total_q, dtype=torch.bool)
    for matchj in range(total_q):
        sj = (matchj // group_size) * group_size
        ej = (matchj // group_size + 1) * group_size
        if sj > 0:
            attn_mask[:, :, matchj, :sj] = True
        if ej < total_q:
            attn_mask[:, :, matchj, ej:] = True
    for match_x in range(total_q):
        if match_x % group_size == 0:
            attn_mask[:, :, match_x, kpt_index] = False
    float_mask = torch.zeros(1, N_HEADS, total_q, total_q, dtype=torch.bfloat16)
    float_mask.masked_fill_(attn_mask, float("-inf"))
    return ttnn.from_torch(float_mask, layout=ttnn.TILE_LAYOUT, device=device,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    import math

    device = ttnn.open_device(device_id=0)
    print("Device opened.")

    full_sd = load_state_dict()
    print(f"Loaded checkpoint: {len(full_sd)} parameters")

    # Build pipeline
    print("Building pipeline...")
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    decoder = TTEDPoseDecoder(device, full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
                              N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES, NUM_BODY_POINTS, NUM_BOX_DEC_LAYERS, NUM_GROUP)
    attn_mask2_tt = prepare_attn_mask2(device)
    print("Pipeline built.")

    # Find a test image
    coco_dir = "/home/yito/datasets/coco"
    import json
    ann_file = os.path.join(coco_dir, "annotations", "person_keypoints_val2017.json")
    with open(ann_file) as f:
        ann = json.load(f)
    img_info = ann["images"][0]
    img_path = os.path.join(coco_dir, "val2017", img_info["file_name"])
    print(f"Test image: {img_info['file_name']}")

    # Run backbone + encoder + two_stage
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
    print("Encoder + two-stage done.")

    # === WARM-UP: run decoder once to JIT-compile all kernels ===
    print("\n--- Warm-up run (JIT compilation) ---")
    with torch.no_grad():
        hs_warmup, refs_warmup = decoder(
            tgt=query_out["tgt"], memory_tt=memory_tt,
            refpoint_embed=query_out["refpoint_embed"],
            spatial_shapes=bb_out["spatial_shapes"],
            level_start_index=bb_out["level_start_index"],
            valid_ratios=bb_out["valid_ratios"],
            memory_key_padding_mask=bb_out["mask_flatten"],
            self_attn_mask=None, self_attn_mask2=attn_mask2_tt,
        )
    print("Warm-up done. All kernels JIT-compiled.")

    # === PROFILED RUN: measure each sub-operation ===
    print("\n--- Profiled run ---")
    timer = Timer(device)

    output = query_out["tgt"].transpose(0, 1)
    reference_points = query_out["refpoint_embed"].transpose(0, 1).sigmoid().clamp(1e-3, 1 - 1e-3)
    valid_ratios = bb_out["valid_ratios"]
    spatial_shapes = bb_out["spatial_shapes"]
    level_start_index = bb_out["level_start_index"]
    mask_flatten = bb_out["mask_flatten"]

    intermediate = []
    tgt_mask = None

    with torch.no_grad():
        for layer_id, layer in enumerate(decoder.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            else:
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]

            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            raw_query_pos = decoder.ref_point_head(query_sine_embed)

            output = profile_decoder_layer(
                timer, layer, output, raw_query_pos, reference_points_input,
                memory_tt, spatial_shapes, level_start_index, mask_flatten,
                tgt_mask, layer_id,
            )
            intermediate.append(decoder.norm(output))

            # Query expansion logic (same as TTEDPoseDecoder)
            if layer_id < NUM_BOX_DEC_LAYERS:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = decoder.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid().clamp(1e-3, 1 - 1e-3)

            if layer_id == NUM_BOX_DEC_LAYERS - 1:
                class_unselected = decoder.class_embed[layer_id](output)
                topk_proposals = torch.topk(class_unselected.max(-1)[0], NUM_GROUP, dim=0)[1]
                new_reference_points_for_box = torch.gather(
                    new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                new_output_for_box = torch.gather(
                    output, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, D_MODEL))
                bs = new_output_for_box.shape[1]
                new_output_for_keypoint = new_output_for_box[:, None, :, :] + \
                    decoder.keypoint_embed.weight[None, :, None, :]
                delta_xy = decoder.pose_embed[-1](new_output_for_keypoint)[..., :2]
                keypoint_xy = (inverse_sigmoid(new_reference_points_for_box[..., :2][:, None]) + delta_xy
                    ).sigmoid().clamp(1e-3, 1 - 1e-3)
                num_queries_box, _, bs_, _ = keypoint_xy.shape
                keypoint_wh_weight = (decoder.hw.weight.unsqueeze(0).unsqueeze(-2)
                    .repeat(num_queries_box, 1, bs_, 1).sigmoid())
                keypoint_wh = keypoint_wh_weight * new_reference_points_for_box[..., 2:][:, None]
                new_reference_points_for_keypoint = torch.cat((keypoint_xy, keypoint_wh), dim=-1)
                new_reference_points = torch.cat(
                    (new_reference_points_for_box.unsqueeze(1), new_reference_points_for_keypoint), dim=1
                ).flatten(0, 1)
                output = torch.cat(
                    (new_output_for_box.unsqueeze(1), new_output_for_keypoint), dim=1
                ).flatten(0, 1)
                tgt_mask = attn_mask2_tt

            if layer_id >= NUM_BOX_DEC_LAYERS:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                output_bbox_norm = output[0::(NUM_BODY_POINTS + 1)]
                reference_before_sigmoid_bbox_norm = reference_before_sigmoid[0::(NUM_BODY_POINTS + 1)]
                delta_unsig_norm = decoder.bbox_embed[layer_id](output_bbox_norm)
                outputs_unsig_norm = delta_unsig_norm + reference_before_sigmoid_bbox_norm
                new_reference_points_for_box_norm = outputs_unsig_norm.sigmoid().clamp(1e-3, 1 - 1e-3)
                kpt_idx = decoder.kpt_index
                output_kpt = output.index_select(0, torch.tensor(kpt_idx, device=output.device))
                delta_xy_unsig = decoder.pose_embed[layer_id - NUM_BOX_DEC_LAYERS](output_kpt)
                outputs_unsig = reference_before_sigmoid.index_select(
                    0, torch.tensor(kpt_idx, device=output.device)).clone()
                delta_hw_unsig = decoder.pose_hw_embed[layer_id - NUM_BOX_DEC_LAYERS](output_kpt)
                outputs_unsig[..., :2] += delta_xy_unsig[..., :2]
                outputs_unsig[..., 2:] += delta_hw_unsig
                new_reference_points_for_keypoint = outputs_unsig.sigmoid().clamp(1e-3, 1 - 1e-3)
                bs_ = new_reference_points_for_box_norm.shape[1]
                new_reference_points_norm = torch.cat((
                    new_reference_points_for_box_norm.unsqueeze(1),
                    new_reference_points_for_keypoint.view(-1, NUM_BODY_POINTS, bs_, 4),
                ), dim=1).flatten(0, 1)
                new_reference_points = new_reference_points_norm

            reference_points = new_reference_points.detach()

    timer.report("Decoder per-op profiling (warm run)")

    # === Overall decoder timing (3 runs) ===
    print("\n--- Overall decoder timing (3 warm runs) ---")
    for run in range(3):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            hs, refs = decoder(
                tgt=query_out["tgt"], memory_tt=memory_tt,
                refpoint_embed=query_out["refpoint_embed"],
                spatial_shapes=bb_out["spatial_shapes"],
                level_start_index=bb_out["level_start_index"],
                valid_ratios=bb_out["valid_ratios"],
                memory_key_padding_mask=bb_out["mask_flatten"],
                self_attn_mask=None, self_attn_mask2=attn_mask2_tt,
            )
        ttnn.synchronize_device(device)
        dt = time.perf_counter() - t0
        print(f"  Run {run+1}: {dt*1000:.1f}ms")

    ttnn.deallocate(memory_tt)
    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
