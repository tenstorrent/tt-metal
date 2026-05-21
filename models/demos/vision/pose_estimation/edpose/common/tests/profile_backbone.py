# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Profile CPU backbone (Swin-L) sub-op timing."""

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/yito/ttwork/tt-metal")
os.environ.setdefault("EDPOSE_ROOT", "/home/yito/ttwork/ED-Pose")

from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import EDPoseBackbone

EDPOSE_ROOT = os.environ["EDPOSE_ROOT"]
CHECKPOINT_PATH = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")


def main():
    print("Loading backbone...")
    backbone = EDPoseBackbone(CHECKPOINT_PATH, device="cpu")
    swin = backbone.swin

    torch.manual_seed(42)
    H, W = 800, 1216
    image_tensor = torch.randn(1, 3, H, W)
    mask = torch.zeros(1, H, W, dtype=torch.bool)

    # Warm-up
    with torch.no_grad():
        _ = backbone(image_tensor, mask)

    # Profile Swin stages
    print("\n=== Profiling Swin-L stages ===")
    with torch.no_grad():
        # PatchEmbed
        t0 = time.perf_counter()
        x = swin.patch_embed(image_tensor)
        t_patch = time.perf_counter() - t0
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        print(f"  PatchEmbed: {t_patch*1000:.1f}ms  shape=({Wh},{Ww}) tokens={Wh*Ww}")

        stage_times = []
        for i in range(swin.num_layers):
            t0 = time.perf_counter()
            layer = swin.layers[i]
            x_out, _H, _W, x, Wh, Ww = layer(x, Wh, Ww)
            t_stage = time.perf_counter() - t0
            stage_times.append(t_stage)

            norm_layer = getattr(swin, f'norm{i}')
            x_out_normed = norm_layer(x_out)
            feat = x_out_normed.view(-1, _H, _W, swin.num_features[i]).permute(0, 3, 1, 2).contiguous()

            nblocks = swin.layers[i].depth
            dim = swin.num_features[i]
            heads = swin.layers[i].blocks[0].num_heads
            print(f"  Stage {i}: {t_stage*1000:.1f}ms  depth={nblocks} dim={dim} heads={heads} "
                  f"out=({_H},{_W}) tokens={_H*_W} feat={feat.shape}")

    # Profile input projection
    print("\n=== Profiling input_proj + position encoding ===")
    with torch.no_grad():
        bb_out = backbone(image_tensor, mask)

    # Detailed backbone call
    with torch.no_grad():
        # Re-run backbone internals
        from util.misc import NestedTensor
        nested = NestedTensor(image_tensor, mask)

        t0 = time.perf_counter()
        features_dict = swin(nested)
        t_swin = time.perf_counter() - t0
        print(f"  Full Swin forward: {t_swin*1000:.1f}ms")

        t0 = time.perf_counter()
        srcs, masks_list, poss = [], [], []
        for lvl in range(len(features_dict)):
            feat = features_dict[lvl]
            src_l, mask_l = feat.tensors, feat.mask
            srcs.append(backbone.input_proj[lvl](src_l))
            masks_list.append(mask_l)
            poss.append(backbone.pos_embed(src_l, mask_l).to(src_l.dtype))
        t_proj = time.perf_counter() - t0
        print(f"  input_proj + pos_embed (4 levels): {t_proj*1000:.1f}ms")

        t0 = time.perf_counter()
        if backbone.num_feature_levels > len(features_dict):
            src_extra = backbone.input_proj[4](features_dict[3].tensors)
            mask_extra = F.interpolate(mask[None].float(), size=src_extra.shape[-2:]).to(torch.bool)[0]
            pos_extra = backbone.pos_embed(src_extra, mask_extra).to(src_extra.dtype)
            srcs.append(src_extra)
            masks_list.append(mask_extra)
            poss.append(pos_extra)
        t_extra = time.perf_counter() - t0
        print(f"  Extra level (stride-2 conv): {t_extra*1000:.1f}ms")

        t0 = time.perf_counter()
        src_flatten, mask_flatten_list, lvl_pos_list = [], [], []
        spatial_shapes = []
        for lvl, (src_l, mask_l, pos_l) in enumerate(zip(srcs, masks_list, poss)):
            bs, c, h, w = src_l.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src_l.flatten(2).transpose(1, 2))
            mask_flatten_list.append(mask_l.flatten(1))
            if backbone.level_embed is not None:
                lvl_pos = pos_l.flatten(2).transpose(1, 2) + backbone.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos = pos_l.flatten(2).transpose(1, 2)
            lvl_pos_list.append(lvl_pos)

        src_flat = torch.cat(src_flatten, 1)
        mask_flat = torch.cat(mask_flatten_list, 1)
        pos_flat = torch.cat(lvl_pos_list, 1)
        t_flatten = time.perf_counter() - t0
        print(f"  Flatten + concat: {t_flatten*1000:.1f}ms")

    # Summary
    total_stages = sum(stage_times)
    print(f"\n=== Summary ===")
    print(f"  PatchEmbed:  {t_patch*1000:>7.1f}ms")
    for i, t in enumerate(stage_times):
        print(f"  Stage {i}:     {t*1000:>7.1f}ms ({t/total_stages*100:.0f}%)")
    print(f"  input_proj:  {t_proj*1000:>7.1f}ms")
    print(f"  Extra level: {t_extra*1000:>7.1f}ms")
    print(f"  Flatten:     {t_flatten*1000:>7.1f}ms")
    print(f"  Total stages:{total_stages*1000:>7.1f}ms")

    # Profile individual block within Stage 2 (dominant)
    print(f"\n=== Stage 2 block-level profile (18 blocks) ===")
    with torch.no_grad():
        # Reset: run up to stage 2
        x = swin.patch_embed(image_tensor)
        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        for i in range(2):
            _, _, _, x, Wh, Ww = swin.layers[i](x, Wh, Ww)

        layer2 = swin.layers[2]
        import numpy as np
        Hp = int(np.ceil(Wh / layer2.window_size)) * layer2.window_size
        Wp = int(np.ceil(Ww / layer2.window_size)) * layer2.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1))
        h_slices = (slice(0, -layer2.window_size),
                    slice(-layer2.window_size, -layer2.shift_size),
                    slice(-layer2.shift_size, None))
        w_slices = (slice(0, -layer2.window_size),
                    slice(-layer2.window_size, -layer2.shift_size),
                    slice(-layer2.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        from models.demos.vision.pose_estimation.edpose.common.tests.profile_backbone import window_partition_ref
        mask_windows = window_partition_ref(img_mask, layer2.window_size)
        mask_windows = mask_windows.view(-1, layer2.window_size * layer2.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        block_times = []
        for j, blk in enumerate(layer2.blocks):
            blk.H, blk.W = Wh, Ww
            t0 = time.perf_counter()
            x = blk(x, attn_mask)
            block_times.append(time.perf_counter() - t0)

        for j, t in enumerate(block_times):
            shift = "shifted" if j % 2 == 1 else "normal"
            print(f"  Block {j:2d} ({shift:7s}): {t*1000:.1f}ms")
        print(f"  Total 18 blocks: {sum(block_times)*1000:.1f}ms")
        print(f"  Avg per block:   {sum(block_times)/len(block_times)*1000:.1f}ms")

    print("\nDone.")


def window_partition_ref(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


if __name__ == "__main__":
    main()
