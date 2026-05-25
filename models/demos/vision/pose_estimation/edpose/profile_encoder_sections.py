# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile encoder layer sections: attention vs FFN vs norms."""

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


def profile_layer_sections(layer, src_tt, pos_tt, bb_out, normalizer,
                           cpu_ref_levels, cpu_norm_levels, device):
    """Profile one encoder layer by section."""
    ttnn.synchronize_device(device)

    # === Section 1: MSDeformAttn (self-attention) ===
    t0 = time.time()
    query = ttnn.add(src_tt, pos_tt)
    src2 = layer.self_attn(
        query, bb_out["reference_points"], src_tt, bb_out["spatial_shapes"],
        bb_out["level_start_index"], bb_out["mask_flatten"],
        normalizer=normalizer,
        cpu_ref_levels=cpu_ref_levels, cpu_norm_levels=cpu_norm_levels,
    )
    ttnn.synchronize_device(device)
    t_attn = time.time() - t0

    # === Section 2: Residual + LayerNorm 1 ===
    t0 = time.time()
    ttnn.deallocate(query)
    old = src_tt
    out = ttnn.add(src_tt, src2)
    ttnn.deallocate(old)
    ttnn.deallocate(src2)
    old = out
    out = ttnn.layer_norm(out, weight=layer.norm1_w, bias=layer.norm1_b)
    ttnn.deallocate(old)
    ttnn.synchronize_device(device)
    t_norm1 = time.time() - t0

    # === Section 3: FFN ===
    t0 = time.time()
    ffn = ttnn.linear(out, layer.ffn1_w, bias=layer.ffn1_b)
    old = ffn
    ffn = ttnn.relu(ffn)
    ttnn.deallocate(old)
    old = ffn
    ffn = ttnn.linear(ffn, layer.ffn2_w, bias=layer.ffn2_b)
    ttnn.deallocate(old)
    ttnn.synchronize_device(device)
    t_ffn = time.time() - t0

    # === Section 4: Residual + LayerNorm 2 ===
    t0 = time.time()
    old = out
    out = ttnn.add(out, ffn)
    ttnn.deallocate(old)
    ttnn.deallocate(ffn)
    old = out
    out = ttnn.layer_norm(out, weight=layer.norm2_w, bias=layer.norm2_b)
    ttnn.deallocate(old)
    ttnn.synchronize_device(device)
    t_norm2 = time.time() - t0

    return out, {
        "attention": t_attn,
        "res+norm1": t_norm1,
        "ffn": t_ffn,
        "res+norm2": t_norm2,
    }


def profile_attn_subsections(attn_module, query_tt, src_tt, reference_points,
                             spatial_shapes, level_start_index, mask_flatten,
                             normalizer, cpu_ref_levels, cpu_norm_levels, device):
    """Profile MSDeformAttn internals without excessive sync barriers."""
    M = attn_module.n_heads
    D = attn_module.d_per_head
    L = attn_module.n_levels
    P = attn_module.n_points

    ttnn.synchronize_device(device)

    # --- Linears ---
    t0 = time.time()
    value_tt = ttnn.linear(src_tt, attn_module.value_proj_w, bias=attn_module.value_proj_b)
    offsets_tt = ttnn.linear(query_tt, attn_module.sampling_offsets_w, bias=attn_module.sampling_offsets_b)
    attn_w_tt = ttnn.linear(query_tt, attn_module.attention_weights_w, bias=attn_module.attention_weights_b)
    ttnn.synchronize_device(device)
    t_linears = time.time() - t0

    # --- Offsets transfer + attn weights (overlapped) ---
    t0 = time.time()
    offsets_t = ttnn.to_torch(offsets_tt).float()
    ttnn.deallocate(offsets_tt)
    N = 1
    Lq = offsets_t.shape[1]
    offsets_t = offsets_t.view(N, Lq, M, L, P, 2)
    t_offsets_cpu = time.time() - t0

    t0 = time.time()
    attn_w_tt, _, _ = attn_module._compute_attn_weights(attn_w_tt, M, L, P)
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

    # --- Per-level: CPU grid compute + from_torch + grid_sample + weighted_sum ---
    t_cpu_grid_total = 0
    t_from_torch_total = 0

    t0_levels = time.time()
    result_accum = None
    for lid, (H, W) in enumerate(spatial_shapes):
        H, W = int(H), int(W)
        start = int(level_start_index[lid])
        end = start + H * W

        val = ttnn.slice(value_tt, [0, start, 0, 0], [N, end, M, D])
        val = ttnn.transpose(val, 1, 2)
        val = ttnn.reshape(val, (N * M, H, W, D))

        tc0 = time.time()
        off_l = offsets_t[:, :, :, lid, :, :]
        locs = cpu_ref_levels[lid] + off_l / cpu_norm_levels[lid]
        grid_l = locs.mul_(2).sub_(1)
        grid_l = grid_l.transpose(1, 2).flatten(0, 1).contiguous().float()

        x = grid_l[..., 0]
        y = grid_l[..., 1]
        h_coord = (y + 1.0) * (H * 0.5) - 0.5
        w_coord = (x + 1.0) * (W * 0.5) - 0.5
        h0 = torch.floor(h_coord)
        w0 = torch.floor(w_coord)
        h_frac = h_coord - h0
        w_frac = w_coord - w0
        h0i = h0.to(torch.int32)
        w0i = w0.to(torch.int32)
        h0v = (h0i >= 0) & (h0i < H)
        h1v = ((h0i + 1) >= 0) & ((h0i + 1) < H)
        w0v = (w0i >= 0) & (w0i < W)
        w1v = ((w0i + 1) >= 0) & ((w0i + 1) < W)
        hfi = 1.0 - h_frac
        wfi = 1.0 - w_frac
        wt_nw = (hfi * wfi * (h0v & w0v).float()).to(torch.bfloat16)
        wt_ne = (hfi * w_frac * (h0v & w1v).float()).to(torch.bfloat16)
        wt_sw = (h_frac * wfi * (h1v & w0v).float()).to(torch.bfloat16)
        wt_se = (h_frac * w_frac * (h1v & w1v).float()).to(torch.bfloat16)
        h0_bf16 = h0.clamp(-32768, 32767).to(torch.int16).view(torch.bfloat16)
        w0_bf16 = w0.clamp(-32768, 32767).to(torch.int16).view(torch.bfloat16)
        packed = torch.stack([h0_bf16, w0_bf16, wt_nw, wt_ne, wt_sw, wt_se], dim=-1)
        packed = packed.reshape(N * M, Lq, 1, 6 * P)
        tc1 = time.time()
        t_cpu_grid_total += (tc1 - tc0)

        tf0 = time.time()
        precomputed_tt = ttnn.from_torch(packed.contiguous(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        tf1 = time.time()
        t_from_torch_total += (tf1 - tf0)

        sampled = ttnn.grid_sample(val, precomputed_tt, use_precomputed_grid=True, align_corners=False)
        ttnn.deallocate(val)
        ttnn.deallocate(precomputed_tt)

        attn_l_tt = ttnn.slice(attn_w_tt, [0, 0, lid * P, 0], [N * M, Lq, (lid + 1) * P, 1])
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
    t_levels_total = time.time() - t0_levels

    # --- Output reshape + proj ---
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
    ttnn.deallocate(attn_w_tt)
    ttnn.deallocate(output)

    return {
        "linears": t_linears,
        "offsets_to_cpu": t_offsets_cpu,
        "attn_weights_device": t_attn_weights,
        "levels_total": t_levels_total,
        "  cpu_grid_compute": t_cpu_grid_total,
        "  from_torch_transfer": t_from_torch_total,
        "  device_ops": t_levels_total - t_cpu_grid_total - t_from_torch_total,
        "output_reshape+proj": t_output,
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

    # Warm-up
    print("Warm-up run...")
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

    normalizer = torch.stack(
        [bb_out["spatial_shapes"][..., 1], bb_out["spatial_shapes"][..., 0]], -1
    ).float()
    L = len(bb_out["spatial_shapes"])
    ref_f = bb_out["reference_points"].float()
    cpu_ref_levels = [ref_f[:, :, lid, :].unsqueeze(2).unsqueeze(3) for lid in range(L)]
    cpu_norm_levels = [normalizer[lid].view(1, 1, 1, 1, 2) for lid in range(L)]

    # === Test 1: Layer section breakdown ===
    N_RUNS = 3
    print(f"=== Layer Section Breakdown ({N_RUNS} runs, layer 0) ===\n")

    for run in range(N_RUNS):
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        out, timings = profile_layer_sections(
            encoder.layers[0], src_tt, pos_tt, bb_out, normalizer,
            cpu_ref_levels, cpu_norm_levels, device)

        total = sum(timings.values())
        print(f"Run {run+1}:")
        for k, v in timings.items():
            print(f"  {k:>15}: {v*1000:>7.1f}ms  ({v/total*100:>5.1f}%)")
        print(f"  {'TOTAL':>15}: {total*1000:>7.1f}ms")
        print()

        ttnn.deallocate(out)
        ttnn.deallocate(pos_tt)

    # === Test 2: Attention subsection breakdown ===
    print(f"=== Attention Subsection Breakdown ({N_RUNS} runs, layer 0) ===\n")

    for run in range(N_RUNS):
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        query = ttnn.add(src_tt, pos_tt)
        attn_timings = profile_attn_subsections(
            encoder.layers[0].self_attn, query, src_tt, bb_out["reference_points"],
            bb_out["spatial_shapes"], bb_out["level_start_index"], bb_out["mask_flatten"],
            normalizer, cpu_ref_levels, cpu_norm_levels, device)
        ttnn.deallocate(query)

        total = sum(v for k, v in attn_timings.items() if not k.startswith("  "))
        print(f"Run {run+1}:")
        for k, v in attn_timings.items():
            if k.startswith("  "):
                print(f"  {k:>25}: {v*1000:>7.1f}ms")
            else:
                print(f"  {k:>25}: {v*1000:>7.1f}ms  ({v/total*100:>5.1f}%)")
        print(f"  {'TOTAL':>25}: {total*1000:>7.1f}ms")
        print()

        ttnn.deallocate(src_tt)
        ttnn.deallocate(pos_tt)

    # === Test 3: Full encoder per-layer timing ===
    print(f"=== Full Encoder Per-Layer Timing ({N_RUNS} runs) ===\n")

    for run in range(N_RUNS):
        with torch.no_grad():
            bb_out = backbone(tensor, mask)
        src_tt = ttnn.from_torch(bb_out["src_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        pos_tt = ttnn.from_torch(bb_out["pos_flatten"].to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        output = src_tt
        layer_times = []
        for i, layer in enumerate(encoder.layers):
            ttnn.synchronize_device(device)
            t0 = time.time()
            output = layer(
                output, pos_tt, bb_out["reference_points"], bb_out["spatial_shapes"],
                bb_out["level_start_index"], bb_out["mask_flatten"],
                normalizer=normalizer,
                cpu_ref_levels=cpu_ref_levels, cpu_norm_levels=cpu_norm_levels,
            )
            ttnn.synchronize_device(device)
            t1 = time.time()
            layer_times.append(t1 - t0)

        total = sum(layer_times)
        print(f"Run {run+1}: ", end="")
        for i, t in enumerate(layer_times):
            print(f"L{i}={t*1000:.0f}", end="  ")
        print(f"total={total*1000:.0f}ms")

        ttnn.deallocate(output)
        ttnn.deallocate(pos_tt)

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
