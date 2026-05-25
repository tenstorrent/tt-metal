# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test: full pipeline x2 with memory stats between runs."""

import os
import sys
import time
import gc

import torch
import torch.nn as nn
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
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_two_stage import TwoStageQueryGenerator
from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import TTEDPoseDecoder

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")
IMAGE_PATH = "/home/yito/datasets/coco/val2017/000000000139.jpg"

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


def prepare_attn_mask2():
    total_q = NUM_GROUP * (NUM_BODY_POINTS + 1)
    group_size = NUM_BODY_POINTS + 1
    kpt_index = [x for x in range(total_q) if x % group_size == 0]
    attn_mask = torch.zeros(1, N_HEADS, total_q, total_q, dtype=torch.bool)
    for matchj in range(total_q):
        sj = (matchj // group_size) * group_size
        ej = sj + group_size
        if sj > 0:
            attn_mask[:, :, matchj, :sj] = True
        if ej < total_q:
            attn_mask[:, :, matchj, ej:] = True
    for match_x in range(total_q):
        if match_x % group_size == 0:
            attn_mask[:, :, match_x, kpt_index] = False
    return attn_mask.flatten(0, 1)


def print_mem(device, label):
    stats = ttnn.device.get_memory_stats(device)
    for bank_id, info in stats.items():
        if "dram" in str(bank_id).lower() or bank_id == 0:
            print(f"  [{label}] bank={bank_id}: {info}")
            break


def main():
    device = ttnn.open_device(device_id=0)
    full_sd = load_state_dict()

    print("Building full pipeline...")
    t0 = time.time()
    backbone = TTSwinLBackbone(device, CHECKPOINT_PATH)
    enc_sd = {k[len("transformer.encoder."):]: v.float()
              for k, v in full_sd.items() if k.startswith("transformer.encoder.layers.")}
    encoder = TTDeformableEncoder(
        device, enc_sd, "layers", N_ENC_LAYERS, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS)
    two_stage = TwoStageQueryGenerator(full_sd, D_MODEL, NUM_QUERIES, NUM_CLASSES)
    decoder = TTEDPoseDecoder(
        device, full_sd, D_MODEL, D_FFN, N_LEVELS, N_HEADS, N_POINTS,
        N_DEC_LAYERS, NUM_QUERIES, NUM_CLASSES, NUM_BODY_POINTS,
        NUM_BOX_DEC_LAYERS, NUM_GROUP)
    attn_mask2 = prepare_attn_mask2()
    print(f"Built in {time.time() - t0:.1f}s")

    # Check DRAM allocation after model load
    try:
        stats = ttnn.device.get_memory_stats(device)
        print(f"Memory stats after model load: {stats}")
    except Exception as e:
        print(f"Cannot get memory stats: {e}")

    # Try to get DRAM info via num_dram_channels / dram_size_per_channel
    try:
        n_ch = device.num_dram_channels()
        print(f"DRAM channels: {n_ch}")
    except Exception:
        pass

    tensor, mask = preprocess_image(IMAGE_PATH)
    print(f"Image padded: {tensor.shape[2]}x{tensor.shape[3]}\n")

    for i in range(2):
        print(f"=== Run {i+1}/2 ===")
        gc.collect()

        t = time.time()
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        print(f"  backbone: {(time.time()-t)*1000:.0f}ms")

        t = time.time()
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        with torch.no_grad():
            enc_out_tt = encoder(src_tt, pos_tt, bb_out["reference_points"],
                bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"])
        memory = ttnn.to_torch(enc_out_tt).float()
        ttnn.deallocate(src_tt)
        ttnn.deallocate(pos_tt)
        print(f"  encoder: {(time.time()-t)*1000:.0f}ms")

        t = time.time()
        with torch.no_grad():
            query_out = two_stage(memory, bb_out["mask_flatten"], bb_out["spatial_shapes"])
        print(f"  two_stage: {(time.time()-t)*1000:.0f}ms")

        gc.collect()
        print(f"  before decoder (gc collected)")

        t = time.time()
        print(f"  starting decoder (layer-by-layer)...")

        # Run decoder manually to find which layer hangs
        from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import inverse_sigmoid

        output = query_out["tgt"].transpose(0, 1)
        reference_points = query_out["refpoint_embed"].transpose(0, 1).sigmoid().clamp(1e-3, 1 - 1e-3)
        valid_ratios = bb_out["valid_ratios"]

        def _to_ttnn_additive_mask(m):
            if m is None:
                return None
            float_mask = torch.zeros_like(m, dtype=torch.bfloat16)
            float_mask.masked_fill_(m, float("-inf"))
            return ttnn.from_torch(
                float_mask, layout=ttnn.TILE_LAYOUT, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        tgt_mask = None
        tgt_mask2 = _to_ttnn_additive_mask(attn_mask2)

        from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_decoder_wrapper import gen_sineembed_for_position

        for layer_id, layer in enumerate(decoder.layers):
            lt = time.time()
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            else:
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :])
            raw_query_pos = decoder.ref_point_head(query_sine_embed)

            output = layer(
                tgt=output,
                tgt_query_pos=raw_query_pos,
                tgt_reference_points=reference_points_input,
                memory=enc_out_tt,
                memory_spatial_shapes=bb_out["spatial_shapes"],
                memory_level_start_index=bb_out["level_start_index"],
                memory_key_padding_mask=bb_out["mask_flatten"],
                self_attn_mask=tgt_mask,
            )
            print(f"    layer {layer_id}: {(time.time()-lt)*1000:.0f}ms, output: {output.shape}")

            # Simplified: skip refinement logic, just move to next layer
            if layer_id < decoder.num_box_decoder_layers:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = decoder.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid().clamp(1e-3, 1 - 1e-3)
                if layer_id == decoder.num_box_decoder_layers - 1:
                    tgt_mask = tgt_mask2
                    # Query expansion
                    class_unselected = decoder.class_embed[layer_id](output)
                    topk_proposals = torch.topk(class_unselected.max(-1)[0], decoder.num_group, dim=0)[1]
                    new_reference_points_for_box = torch.gather(
                        new_reference_points, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                    new_output_for_box = torch.gather(
                        output, 0, topk_proposals.unsqueeze(-1).repeat(1, 1, decoder.d_model))
                    new_output_for_keypoint = new_output_for_box[:, None, :, :] + \
                        decoder.keypoint_embed.weight[None, :, None, :]
                    delta_xy = decoder.pose_embed[-1](new_output_for_keypoint)[..., :2]
                    keypoint_xy = (
                        inverse_sigmoid(new_reference_points_for_box[..., :2][:, None]) + delta_xy
                    ).sigmoid().clamp(1e-3, 1 - 1e-3)
                    num_queries_box = keypoint_xy.shape[0]
                    bs_ = keypoint_xy.shape[2]
                    keypoint_wh_weight = (
                        decoder.hw.weight.unsqueeze(0).unsqueeze(-2)
                        .repeat(num_queries_box, 1, bs_, 1).sigmoid())
                    keypoint_wh = keypoint_wh_weight * new_reference_points_for_box[..., 2:][:, None]
                    new_reference_points_for_keypoint = torch.cat((keypoint_xy, keypoint_wh), dim=-1)
                    new_reference_points = torch.cat(
                        (new_reference_points_for_box.unsqueeze(1), new_reference_points_for_keypoint), dim=1
                    ).flatten(0, 1)
                    output = torch.cat(
                        (new_output_for_box.unsqueeze(1), new_output_for_keypoint), dim=1
                    ).flatten(0, 1)
            else:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                output_bbox_norm = output[0::(NUM_BODY_POINTS + 1)]
                reference_before_sigmoid_bbox_norm = reference_before_sigmoid[0::(NUM_BODY_POINTS + 1)]
                delta_unsig_norm = decoder.bbox_embed[layer_id](output_bbox_norm)
                outputs_unsig_norm = delta_unsig_norm + reference_before_sigmoid_bbox_norm
                new_reference_points_for_box_norm = outputs_unsig_norm.sigmoid().clamp(1e-3, 1 - 1e-3)
                kpt_index = decoder.kpt_index
                output_kpt = output.index_select(0, torch.tensor(kpt_index))
                delta_xy_unsig = decoder.pose_embed[layer_id - decoder.num_box_decoder_layers](output_kpt)
                outputs_unsig = reference_before_sigmoid.index_select(0, torch.tensor(kpt_index)).clone()
                delta_hw_unsig = decoder.pose_hw_embed[layer_id - decoder.num_box_decoder_layers](output_kpt)
                outputs_unsig[..., :2] += delta_xy_unsig[..., :2]
                outputs_unsig[..., 2:] += delta_hw_unsig
                new_reference_points_for_keypoint = outputs_unsig.sigmoid().clamp(1e-3, 1 - 1e-3)
                bs_ = new_reference_points_for_box_norm.shape[1]
                new_reference_points = torch.cat(
                    (new_reference_points_for_box_norm.unsqueeze(1),
                     new_reference_points_for_keypoint.view(-1, NUM_BODY_POINTS, bs_, 4)),
                    dim=1).flatten(0, 1)

            reference_points = new_reference_points.detach()

        if tgt_mask2 is not None and isinstance(tgt_mask2, ttnn.Tensor):
            ttnn.deallocate(tgt_mask2)
        ttnn.deallocate(enc_out_tt)
        print(f"  decoder total: {(time.time()-t)*1000:.0f}ms")
        print(f"  Run {i+1} complete\n")

    ttnn.close_device(device)
    print("Done.")


if __name__ == "__main__":
    main()
