# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# HybridEncoder: AIFI on the coarsest scale + CCFM neck.

import torch
import ttnn
import math

from tt.rtdetr_encoder import run_aifi
from tt.resnet_blocks import conv_block

_EMBED_DIM = 256

def _sinusoidal_pos_embed(h, w, embed_dim, device):
    """Generates 2D sincos pos embeddings exactly matching Lyuwenyu's block-concat math."""
    grid_w = torch.arange(int(w), dtype=torch.float32)
    grid_h = torch.arange(int(h), dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (10000.0 ** omega)

    out_w = grid_w.flatten()[..., None] @ omega[None]
    out_h = grid_h.flatten()[..., None] @ omega[None]

    # Block concatenation
    pe = torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)
    pe = pe.reshape(1, 1, h * w, embed_dim)

    return ttnn.from_torch(
        pe, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        device=device, memory_config=ttnn.L1_MEMORY_CONFIG,
    )

def _upsample_concat(x_fine, x_coarse, device, h_out, w_out, c):
    """Pure on-device upsample avoiding TILE_LAYOUT offset corruption."""
    batch_size = x_coarse.shape[0]
    h_in = h_out // 2
    w_in = w_out // 2
    seq_len = h_in * w_in

    # 1. Un-tile to strip physical hardware padding
    x_coarse_rm = ttnn.to_layout(x_coarse, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    
    # 2. Slice out the hardware padding
    x_coarse_sliced = ttnn.slice(
        x_coarse_rm, 
        [0, 0, 0, 0], 
        [batch_size, 1, seq_len, c]
    )

    # 3. Reshape the logical view to a 4D spatial grid
    x_coarse_4d = ttnn.reshape(x_coarse_sliced, (batch_size, h_in, w_in, c))

    # 4. Native distributed nearest-neighbor interpolation
    x_up_4d = ttnn.upsample(x_coarse_4d, 2, mode="nearest", memory_config=ttnn.L1_MEMORY_CONFIG)

    # Now that data is safely in a new buffer, we can garbage collect the original un-tiled buffer
    ttnn.deallocate(x_coarse_rm)

    # 5. Reshape and re-tile for downstream CSP layers
    x_up_flat = ttnn.reshape(x_up_4d, (batch_size, 1, h_out * w_out, c))
    x_up_tiled = ttnn.to_layout(x_up_flat, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    
    # ttnn.to_layout allocates new memory, so the 4D upsample buffer can be freed
    ttnn.deallocate(x_up_4d)

    # 6. SRAM Concatenation
    out = ttnn.concat([x_up_tiled, x_fine], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x_up_tiled)
    
    return out

def _csp_rep_layer(x, params, device, h, w):
    x1_pre, _ = conv_block(x, params.conv1, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
    x1 = ttnn.silu(x1_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x1_pre)

    for bn in params.bottlenecks:
        out1, _ = conv_block(x1, bn.conv1, device, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), input_height=h, input_width=w, activation="")
        out2, _ = conv_block(x1, bn.conv2, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
        out_add = ttnn.add(out1, out2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(out1)
        ttnn.deallocate(out2)
        
        x1 = ttnn.silu(out_add, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(out_add)

    x2_pre, _ = conv_block(x, params.conv2, device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h, input_width=w, activation="")
    x2 = ttnn.silu(x2_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x2_pre)
    out = ttnn.add(x1, x2, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(x1)
    ttnn.deallocate(x2)
    
    return out, h, w

def hybrid_encoder(s3, s4, s5, params, device, return_debug=False):
    h3, w3 = 80, 80
    h4, w4 = 40, 40
    h5, w5 = 20, 20

    p3, _ = conv_block(s3, params.input_proj[0], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h3, input_width=w3, activation="")
    p4, _ = conv_block(s4, params.input_proj[1], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h4, input_width=w4, activation="")
    p5, _ = conv_block(s5, params.input_proj[2], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h5, input_width=w5, activation="")

    # Capture pre-AIFI p5
    pre_aifi_p5_tt = ttnn.clone(p5, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_debug else None

    # AIFI — run directly on 400 tokens, no padding
    pos5 = _sinusoidal_pos_embed(h5, w5, _EMBED_DIM, device)
    p5_out = run_aifi(p5, params.encoder_layers, device, pos_embed=pos5)
    ttnn.deallocate(pos5)
    ttnn.deallocate(p5)

    # Capture post-AIFI p5
    post_aifi_p5_tt = ttnn.clone(p5_out, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_debug else None

    # FPN path
    p5_lat_pre, _ = conv_block(p5_out, params.lateral_convs[0], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h5, input_width=w5, activation="")
    p5_lat = ttnn.silu(p5_lat_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(p5_lat_pre)

    # Capture post-lateral-conv p5
    p5_lat_debug = ttnn.clone(p5_lat, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_debug else None

    p4_cat = _upsample_concat(p4, p5_lat, device, h4, w4, _EMBED_DIM)

    # Capture p4 after upsample+concat (before fpn_block[0])
    p4_cat_debug = ttnn.clone(p4_cat, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_debug else None

    p4_td, _, _ = _csp_rep_layer(p4_cat, params.fpn_blocks[0], device, h4, w4)
    ttnn.deallocate(p4_cat)

    # Capture p4 after fpn_block[0]
    p4_td_debug = ttnn.clone(p4_td, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_debug else None

    p4_lat_pre, _ = conv_block(p4_td, params.lateral_convs[1], device, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), input_height=h4, input_width=w4, activation="")
    p4_lat = ttnn.silu(p4_lat_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(p4_lat_pre)

    # Capture post-lateral-conv p4
    p4_lat_debug = ttnn.clone(p4_lat, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_debug else None

    p3_cat = _upsample_concat(p3, p4_lat, device, h3, w3, _EMBED_DIM)

    # Capture p3 after upsample+concat (before fpn_block[1])
    p3_cat_debug = ttnn.clone(p3_cat, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_debug else None

    p3_out, _, _ = _csp_rep_layer(p3_cat, params.fpn_blocks[1], device, h3, w3)
    ttnn.deallocate(p3_cat)

    # Capture p3 after fpn_block[1]
    p3_out_debug = ttnn.clone(p3_out, memory_config=ttnn.DRAM_MEMORY_CONFIG) if return_debug else None

    # PAN path
    p3_down_pre, _ = conv_block(p3_out, params.downsample_convs[0], device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), input_height=h3, input_width=w3, activation="")
    p3_down = ttnn.silu(p3_down_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(p3_down_pre)

    p4_cat2 = ttnn.concat([p3_down, p4_lat], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    p4_out, _, _ = _csp_rep_layer(p4_cat2, params.pan_blocks[0], device, h4, w4)
    ttnn.deallocate(p3_down)
    ttnn.deallocate(p4_lat)
    ttnn.deallocate(p4_cat2)

    p4_down_pre, _ = conv_block(p4_out, params.downsample_convs[1], device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), input_height=h4, input_width=w4, activation="")
    p4_down = ttnn.silu(p4_down_pre, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(p4_down_pre)

    p5_cat2 = ttnn.concat([p4_down, p5_lat], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    p5_out_final, _, _ = _csp_rep_layer(p5_cat2, params.pan_blocks[1], device, h5, w5)
    ttnn.deallocate(p4_down)
    ttnn.deallocate(p5_lat)
    ttnn.deallocate(p5_cat2)

    if return_debug:
        debug_tensors = {
            "pre_aifi_p5":  pre_aifi_p5_tt,
            "post_aifi_p5": post_aifi_p5_tt,
            "p5_lat":       p5_lat_debug,
            "p4_cat":       p4_cat_debug,
            "p4_td":        p4_td_debug,
            "p4_lat":       p4_lat_debug,
            "p3_cat":       p3_cat_debug,
            "p3_out":       p3_out_debug,
        }
        return [p3_out, p4_out, p5_out_final], debug_tensors
    return [p3_out, p4_out, p5_out_final]

    

