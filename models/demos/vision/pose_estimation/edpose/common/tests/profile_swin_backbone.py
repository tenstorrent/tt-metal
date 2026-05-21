# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile TTSwinBackbone sub-op timing."""

import os
import sys
import time

import torch

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

import ttnn

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_swin_backbone import TTSwinBackbone

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")


def main():
    device = ttnn.open_device(device_id=0)

    print("Loading TTSwinBackbone...")
    bb = TTSwinBackbone(device, CHECKPOINT_PATH)

    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    # Warm-up
    print("Warm-up...")
    with torch.no_grad():
        _ = bb(image_tensor, mask)

    # Profile major sections
    print("\n=== Profiling TTSwinBackbone ===")

    with torch.no_grad():
        # PatchEmbed (CPU)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        x_cpu = bb.patch_embed[0](image_tensor)
        Wh, Ww = x_cpu.size(2), x_cpu.size(3)
        x_cpu = x_cpu.flatten(2).transpose(1, 2)
        x_cpu = bb.patch_norm(x_cpu)
        t_patch = time.perf_counter() - t0
        print(f"  PatchEmbed (CPU): {t_patch*1000:.1f}ms")

        # To device
        t0 = time.perf_counter()
        x_tt = ttnn.from_torch(
            x_cpu.to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(device)
        t_to_dev = time.perf_counter() - t0
        print(f"  PatchEmbed to device: {t_to_dev*1000:.1f}ms")

        # Stages
        stage_times = []
        stage_block_times = []
        Hs, Ws = Wh, Ww
        for si in range(4):
            layer = bb.layers[si]
            ws = layer.window_size
            depth = layer.depth

            # Compute attention mask once
            import numpy as np
            Hp = int(np.ceil(Hs / ws)) * ws
            Wp = int(np.ceil(Ws / ws)) * ws

            ttnn.synchronize_device(device)
            t_stage_start = time.perf_counter()

            # Mask computation
            attn_mask_tt = layer._compute_attn_mask(Hs, Ws)

            blk_times = {"ln1": [], "to_host_part": [], "from_host_part": [],
                         "attn_qkv": [], "attn_host": [], "attn_to_dev": [],
                         "attn_proj": [], "to_host_rev": [], "from_host_rev": [],
                         "residual1": [], "ln2": [], "mlp": [], "residual2": []}

            for bi, blk in enumerate(layer.blocks):
                use_mask = blk.shift_size > 0

                # LN1
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                shortcut = x_tt
                x = ttnn.layer_norm(x_tt, weight=blk.norm1_w, bias=blk.norm1_b)
                ttnn.synchronize_device(device)
                blk_times["ln1"].append(time.perf_counter() - t0)

                # To host + window partition
                t0 = time.perf_counter()
                x_t = ttnn.to_torch(x).float()
                B = x_t.shape[0]
                C = blk.dim
                x_t = x_t[:, :Hs * Ws, :C]
                x_t = x_t.view(B, Hs, Ws, C)
                pad_b = (ws - Hs % ws) % ws
                pad_r = (ws - Ws % ws) % ws
                if pad_b > 0 or pad_r > 0:
                    import torch.nn.functional as Fp
                    x_t = Fp.pad(x_t, (0, 0, 0, pad_r, 0, pad_b))
                _, Hp_cur, Wp_cur, _ = x_t.shape
                if blk.shift_size > 0:
                    x_t = torch.roll(x_t, shifts=(-blk.shift_size, -blk.shift_size), dims=(1, 2))
                nH_win = Hp_cur // ws
                nW_win = Wp_cur // ws
                nW_total = nH_win * nW_win
                x_t = x_t.view(B, nH_win, ws, nW_win, ws, C)
                x_t = x_t.permute(0, 1, 3, 2, 4, 5).contiguous()
                x_t = x_t.view(B * nW_total, ws * ws, C)
                blk_times["to_host_part"].append(time.perf_counter() - t0)

                # From host (window data)
                t0 = time.perf_counter()
                x_win_tt = ttnn.from_torch(
                    x_t.to(torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.synchronize_device(device)
                blk_times["from_host_part"].append(time.perf_counter() - t0)

                # Attention: QKV linear on device
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                qkv = ttnn.linear(x_win_tt, blk.attn.qkv_w, bias=blk.attn.qkv_b)
                ttnn.synchronize_device(device)
                blk_times["attn_qkv"].append(time.perf_counter() - t0)

                # Attention: host math
                t0 = time.perf_counter()
                nH = blk.attn.num_heads
                D = blk.attn.head_dim
                N = ws * ws
                qkv_t = ttnn.to_torch(qkv).float()
                ttnn.deallocate(qkv)
                B_ = qkv_t.shape[0]
                qkv_t = qkv_t[:, :N, :]
                qkv_t = qkv_t.reshape(B_, N, 3, nH, D).permute(2, 0, 3, 1, 4)
                q, k, v = qkv_t[0], qkv_t[1], qkv_t[2]
                q = q * blk.attn.scale
                attn = q @ k.transpose(-2, -1)
                rpb_t = ttnn.to_torch(blk.attn.rel_pos_bias).float()
                rpb_t = rpb_t[:, :nH, :N, :N]
                attn = attn + rpb_t
                if use_mask:
                    mask_t = ttnn.to_torch(attn_mask_tt).float()
                    mask_t = mask_t[:, :, :N, :N]
                    B_per_batch = B_ // nW_total
                    attn = attn.view(B_per_batch, nW_total, nH, N, N) + mask_t.unsqueeze(0)
                    attn = attn.view(B_, nH, N, N)
                attn = torch.softmax(attn, dim=-1)
                out_t = (attn @ v).transpose(1, 2).reshape(B_, N, C)
                blk_times["attn_host"].append(time.perf_counter() - t0)

                # Attention: output back to device + proj
                t0 = time.perf_counter()
                out_tt = ttnn.from_torch(
                    out_t.to(torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.synchronize_device(device)
                blk_times["attn_to_dev"].append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                out_tt = ttnn.linear(out_tt, blk.attn.proj_w, bias=blk.attn.proj_b)
                ttnn.synchronize_device(device)
                blk_times["attn_proj"].append(time.perf_counter() - t0)

                ttnn.deallocate(x_win_tt)

                # To host: window reverse
                t0 = time.perf_counter()
                rev_t = ttnn.to_torch(out_tt).float()
                ttnn.deallocate(out_tt)
                rev_t = rev_t[:, :N, :C]
                rev_t = rev_t.view(B * nW_total, ws, ws, C)
                rev_t = rev_t.view(B, nH_win, nW_win, ws, ws, C)
                rev_t = rev_t.permute(0, 1, 3, 2, 4, 5).contiguous()
                rev_t = rev_t.view(B, Hp_cur, Wp_cur, C)
                if blk.shift_size > 0:
                    rev_t = torch.roll(rev_t, shifts=(blk.shift_size, blk.shift_size), dims=(1, 2))
                if pad_b > 0 or pad_r > 0:
                    rev_t = rev_t[:, :Hs, :Ws, :].contiguous()
                rev_t = rev_t.view(B, Hs * Ws, C)
                blk_times["to_host_rev"].append(time.perf_counter() - t0)

                # From host: back to device
                t0 = time.perf_counter()
                attn_result = ttnn.from_torch(
                    rev_t.to(torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT, device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.synchronize_device(device)
                blk_times["from_host_rev"].append(time.perf_counter() - t0)

                # Residual 1
                t0 = time.perf_counter()
                x_tt = ttnn.add(shortcut, attn_result)
                ttnn.deallocate(attn_result)
                ttnn.deallocate(shortcut)
                ttnn.synchronize_device(device)
                blk_times["residual1"].append(time.perf_counter() - t0)

                # LN2
                t0 = time.perf_counter()
                shortcut2 = x_tt
                x_ln2 = ttnn.layer_norm(x_tt, weight=blk.norm2_w, bias=blk.norm2_b)
                ttnn.synchronize_device(device)
                blk_times["ln2"].append(time.perf_counter() - t0)

                # MLP
                t0 = time.perf_counter()
                x_mlp = ttnn.linear(x_ln2, blk.fc1_w, bias=blk.fc1_b)
                x_mlp = ttnn.gelu(x_mlp)
                x_mlp = ttnn.linear(x_mlp, blk.fc2_w, bias=blk.fc2_b)
                ttnn.synchronize_device(device)
                blk_times["mlp"].append(time.perf_counter() - t0)

                # Residual 2
                t0 = time.perf_counter()
                x_tt = ttnn.add(shortcut2, x_mlp)
                ttnn.deallocate(shortcut2)
                ttnn.synchronize_device(device)
                blk_times["residual2"].append(time.perf_counter() - t0)

            ttnn.deallocate(attn_mask_tt)

            # Output norm
            norm_w, norm_b = bb.output_norms[si]
            x_normed = ttnn.layer_norm(x_tt, weight=norm_w, bias=norm_b)

            # Downsample
            if bb.layers[si].downsample is not None:
                x_down, Hs, Ws = bb.layers[si].downsample(x_tt, Hs, Ws)
                ttnn.deallocate(x_normed)  # we just needed x_out for norm
                x_normed_dummy = None
                x_tt = x_down
            else:
                ttnn.deallocate(x_tt)
                x_tt = x_normed

            ttnn.synchronize_device(device)
            t_stage = time.perf_counter() - t_stage_start
            stage_times.append(t_stage)

            dim_l = bb.num_features[si]
            print(f"\n  Stage {si} ({depth} blocks, dim={dim_l}): {t_stage*1000:.0f}ms")
            for k, vals in blk_times.items():
                total = sum(vals) * 1000
                avg = total / len(vals) if vals else 0
                print(f"    {k:>16s}: total={total:7.1f}ms  avg={avg:5.1f}ms")

        ttnn.deallocate(x_tt)

        # Input proj + pos embed (CPU)
        t0 = time.perf_counter()
        # ... simplified: just measure total time
        print(f"\n  Total stages: {sum(stage_times)*1000:.0f}ms")

    ttnn.close_device(device)
    print("\nDone.")


if __name__ == "__main__":
    main()
